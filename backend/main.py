from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple

import os
import logging
import aiofiles
import polars as pl
import gcsfs

from langchain.schema import Document,BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from sentence_transformers import CrossEncoder

from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import MultifieldParser
from tqdm import tqdm
import faiss

from functools import lru_cache
# ---------------------------------------------------------------------------- #
#                                   Settings                                   #
# ---------------------------------------------------------------------------- #
# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    # Parquet + Whoosh/FAISS
    parquet_path: str = "gs://mda_eu_project/data/consolidated_clean_pred.parquet"
    whoosh_dir:    str = "gs://mda_eu_project/whoosh_index"
    vectorstore_path: str = "gs://mda_eu_project/vectorstore_index"
    # Models
    embedding_model:     str = "sentence-transformers/LaBSE"
    llm_model:           str = "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"
    cross_encoder_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # RAG parameters
    chunk_size:    int = 750
    chunk_overlap: int = 100
    hybrid_k:      int = 50
    assistant_role: str = (
        "You are a concise, factual assistant. Cite Document [ID] for each claim."
    )
    skip_warmup: bool = False
    allowed_origins: List[str] = ["*"]

    class Config:
        env_file = ".env"

settings = Settings()

from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("sentence-transformers/LaBSE")
print(cfg.model_type)

# Pre‐instantiate embedding model (used by filter/compressor)
EMBEDDING = HuggingFaceEmbeddings(model_name=settings.embedding_model,
    model_kwargs={"trust_remote_code": True})

@lru_cache(maxsize=256)
def embed_query_cached(query: str) -> List[float]:
    """Cache embedding vectors for queries."""
    return EMBEDDING.embed_query(query.strip().lower())

# === Whoosh Cache & Builder ===
_WHOOSH_CACHE: Dict[str, index.Index] = {}

async def build_whoosh_index(docs: List[Document], whoosh_dir: str) -> index.Index:
    key = whoosh_dir
    fs = gcsfs.GCSFileSystem()
    local_dir = key
    is_gcs = key.startswith("gs://")
    try:
        # stage local copy for GCS
        if is_gcs:
            local_dir = "/tmp/whoosh_index"
            if not os.path.exists(local_dir):
                if await run_in_threadpool(fs.exists, key):
                    await run_in_threadpool(fs.get, key, local_dir, recursive=True)
                else:
                    os.makedirs(local_dir, exist_ok=True)
        # build once
        if key not in _WHOOSH_CACHE:
            os.makedirs(local_dir, exist_ok=True)
            schema = Schema(
                id=ID(stored=True, unique=True),
                content=TEXT(analyzer=StemmingAnalyzer()),
            )
            ix = index.create_in(local_dir, schema)
            with ix.writer() as writer:
                for doc in docs:
                    writer.add_document(
                        id=doc.metadata.get("id", ""),
                        content=doc.page_content,
                    )
            # push back to GCS atomically
            if is_gcs:
                await run_in_threadpool(fs.put, local_dir, key, recursive=True)
            _WHOOSH_CACHE[key] = ix
        return _WHOOSH_CACHE[key]
    except Exception as e:
        logger.error(f"Failed to build Whoosh index: {e}")
        raise

# === Document Loader ===
async def load_documents(
    path: str,
    sample_size: Optional[int] = None
) -> List[Document]:
    """
    Load a Parquet file from local or GCS, convert to a list of Documents.
    """
    def _read_local(p: str, n: Optional[int]):
        # streaming scan keeps memory low
        lf = pl.scan_parquet(p)
        if n:
            lf = lf.limit(n)
        return lf.collect(streaming=True)

    def _read_gcs(p: str, n: Optional[int]):
        # download to a temp file synchronously, then read with Polars
        fs = gcsfs.GCSFileSystem()
        with tempfile.TemporaryDirectory() as td:
            local_path = os.path.join(td, "data.parquet")
            fs.get(p, local_path, recursive=False)
            df = pl.read_parquet(local_path)
        if n:
            df = df.head(n)
        return df

    try:
        if path.startswith("gs://"):
            df = await run_in_threadpool(_read_gcs, path, sample_size)
        else:
            df = await run_in_threadpool(_read_local, path, sample_size)
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise HTTPException(status_code=500, detail="Document loading failed.")

    docs: List[Document] = []
    for row in df.rows(named=True):
        context_parts: List[str] = []
        # build metadata context
        max_contrib = row.get("ecMaxContribution", "")
        end_date = row.get("endDate", "")
        duration = row.get("durationDays", "")
        status = row.get("status", "")
        legal = row.get("legalBasis", "")
        framework = row.get("frameworkProgramme", "")
        scheme = row.get("fundingScheme", "")
        names = row.get("list_name", []) or []
        cities = row.get("list_city", []) or []
        countries = row.get("list_country", []) or []
        activity = row.get("list_activityType", []) or []
        contributions = row.get("list_ecContribution", []) or []
        smes = row.get("list_sme", []) or []
        project_id =row.get("id", "")
        pred=row.get("predicted_label", "")
        proba=row.get("predicted_prob", "")
        top1_feats=row.get("top1_features", "")
        top2_feats=row.get("top2_features", "")
        top3_feats=row.get("top3_features", "")
        top1_shap=row.get("top1_shap", "")
        top2_shap=row.get("top2_shap", "")
        top3_shap=row.get("top3_shap", "")


        context_parts.append(
            f"This project under framework {framework} with funding scheme {scheme}, status {status}, legal basis {legal}."
        )
        context_parts.append(
            f"It ends on {end_date} after {duration} days and has a max EC contribution of {max_contrib}."
        )
        context_parts.append("Participating organizations:")
        for i, name in enumerate(names):
            city = cities[i] if i < len(cities) else ""
            country = countries[i] if i < len(countries) else ""
            act = activity[i] if i < len(activity) else ""
            contrib = contributions[i] if i < len(contributions) else ""
            sme_flag = "SME" if (smes and i < len(smes) and smes[i]) else "non-SME"
            context_parts.append(
                f"- {name} in {city}, {country}, activity: {act}, contributed: {contrib}, {sme_flag}."
            )
            if status in (None,"signed","SIGNED","Signed"):
                if int(pred) == 1:
                    label = "TERMINATED"
                    score = float(proba)
                else:
                    label = "CLOSED"
                    score = 1 - float(proba)

                score_str = f"{score:.2f}"

                context_parts.append(
                    f"- Project {project_id} is predicted to be {label} (score={score_str}). "
                    f"The 3 most predictive features were: "
                    f"{top1_feats} ({top1_shap:.3f}), "
                    f"{top2_feats} ({top2_shap:.3f}), "
                    f"{top3_feats} ({top3_shap:.3f})."
                )

        title_report = row.get("list_title_report", "")
        objective = row.get("objective", "")
        full_body = f"{title_report} {objective}"
        full_text = " ".join(context_parts + [full_body])
        meta: Dict[str, Any] = {"id": str(row.get("id", "")),"startDate": str(row.get("startDate", "")),"endDate": str(row.get("endDate", "")),"status":str(row.get("status", "")),"legalBasis":str(row.get("legalBasis",""))}
        meta.update({"id": str(row.get("id", "")),"startDate": str(row.get("startDate", "")),"endDate": str(row.get("endDate", "")),"status":str(row.get("status", "")),"legalBasis":str(row.get("legalBasis",""))})
        docs.append(Document(page_content=full_text, metadata=meta))
    return docs

# === BM25 Search ===
async def bm25_search(ix: index.Index, query: str, k: int) -> List[Document]:
    parser = MultifieldParser(["content"], schema=ix.schema)
    def _search() -> List[Document]:
        with ix.searcher() as searcher:
            hits = searcher.search(parser.parse(query), limit=k)
            return [Document(page_content=h["content"], metadata={"id": h["id"]}) for h in hits]
    return await run_in_threadpool(_search)

# === Helper: build or load FAISS with mmap ===
async def build_or_load_faiss(
    chunks: List[Document],
    vectorstore_path: str,
    batch_size: int = 15000
) -> FAISS:
    faiss_index_file = os.path.join(vectorstore_path, "index.faiss")
    # If on-disk exists: memory-map the FAISS index and load metadata separately
    if os.path.exists(faiss_index_file):
        logger.info("Memory-mapping existing FAISS index...")
        mmap_idx = faiss.read_index(faiss_index_file, faiss.IO_FLAG_MMAP)
        # Manually load metadata (docstore and index_to_docstore) without loading the index
        import pickle
        for meta_file in ["faiss.pkl", "index.pkl"]:
            meta_path = os.path.join(vectorstore_path, meta_file)
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    saved = pickle.load(f)
                break
        else:
            raise FileNotFoundError(
                f"Could not find FAISS metadata pickle in {vectorstore_path}"
            )
                # extract metadata
        if isinstance(saved, tuple):
            # Handle metadata tuple of length 2 or 3
            if len(saved) == 3:
                _, docstore, index_to_docstore = saved
            elif len(saved) == 2:
                docstore, index_to_docstore = saved
            else:
                raise ValueError(f"Unexpected metadata tuple length: {len(saved)}")
        else:
            if hasattr(saved, 'docstore'):
                docstore = saved.docstore
            elif hasattr(saved, '_docstore'):
                docstore = saved._docstore
            else:
                raise AttributeError("Could not find docstore in FAISS metadata")
            if hasattr(saved, 'index_to_docstore'):
                index_to_docstore = saved.index_to_docstore
            elif hasattr(saved, '_index_to_docstore'):
                index_to_docstore = saved._index_to_docstore
            elif hasattr(saved, '_faiss_index_to_docstore'):
                index_to_docstore = saved._faiss_index_to_docstore
            else:
                raise AttributeError("Could not find index_to_docstore in FAISS metadata")
        # reconstruct FAISS wrapper
        vs = FAISS(
            embedding_function=EMBEDDING,
            index=mmap_idx,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore,
        )
        return vs

    # 2) Else: build from scratch in batches
    logger.info(f"Building FAISS index in batches of {batch_size}…")
    vs: Optional[FAISS] = None
    for i in tqdm(range(0, len(chunks), batch_size),
                  desc="Building FAISS index",
                  unit="batch"):
        batch = chunks[i : i + batch_size]

        if vs is None:
            vs = FAISS.from_documents(batch, EMBEDDING)
        else:
            vs.add_documents(batch)

        # periodic save every 5 batches
        if (i // batch_size) % 5 == 0:
            vs.save_local(vectorstore_path)

        logger.info(f"  • Saved batch up to document {i + len(batch)} / {len(chunks)}")
    assert vs is not None, "No documents to index!"
    return vs

# === Index Builder ===
async def build_indexes(
    parquet_path: str,
    vectorstore_path: str,
    whoosh_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    debug_size: Optional[int]
) -> Tuple[FAISS, index.Index]:
    docs = await load_documents(parquet_path, debug_size)
    ix = await build_whoosh_index(docs, whoosh_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    # build or load (with mmap) FAISS
    vs = await build_or_load_faiss(chunks, vectorstore_path)

    return vs, ix

# === Hybrid Retriever ===
class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining BM25 and FAISS with cross-encoder re-ranking."""
    # store FAISS and Whoosh under private attributes to avoid Pydantic field errors
    from pydantic import PrivateAttr
    _vs: FAISS = PrivateAttr()
    _ix: index.Index = PrivateAttr()
    _compressor: DocumentCompressorPipeline = PrivateAttr()
    _cross_encoder: CrossEncoder = PrivateAttr()

    def __init__(
        self,
        vs: FAISS,
        ix: index.Index,
        compressor: DocumentCompressorPipeline,
        cross_encoder: CrossEncoder
    ) -> None:
        super().__init__()
        object.__setattr__(self, '_vs', vs)
        object.__setattr__(self, '_ix', ix)
        object.__setattr__(self, '_compressor', compressor)
        object.__setattr__(self, '_cross_encoder', cross_encoder)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # BM25 retrieval using Whoosh index
        bm_docs = await bm25_search(self._ix, query, settings.hybrid_k)
        # Dense retrieval using FAISS
        dense_docs = self._vs.similarity_search_by_vector(
            embed_query_cached(query), k=settings.hybrid_k
        )
        # Cross-encoder re-ranking
        candidates = bm_docs + dense_docs
        scores = self._cross_encoder.predict([
            (query, doc.page_content) for doc in candidates
        ])
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top = [doc for _, doc in ranked[: settings.hybrid_k]]
        # Compress and return
        return self._compressor.compress_documents(top, query=query)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._aget_relevant_documents(query)
        )

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Romain\OneDrive - KU Leuven\focal-pager-460414-e9-45369b738be0.json"
# ---------------------------------------------------------------------------- #
#                               Single Lifespan                                 #
# ---------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # --- 1) RAG Initialization --- #
    logger = logging.getLogger("uvicorn")
    logger.info("Initializing RAG components…")

    # Compressor pipeline to de‐duplicate via embeddings
    compressor = DocumentCompressorPipeline(
        transformers=[EmbeddingsRedundantFilter(embeddings=EMBEDDING)]
    )

    # Cross‐encoder ranker
    cross_encoder = CrossEncoder(settings.cross_encoder_model)

    # Causal LLM pipeline
    llm_model = AutoModelForCausalLM.from_pretrained(settings.llm_model)
    gen_pipe = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=AutoTokenizer.from_pretrained(settings.llm_model),
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
    )
    llm = HuggingFacePipeline(pipeline=gen_pipe)

    # Conversational memory
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

    # Build or load FAISS & Whoosh once
    vs, ix = await build_indexes(
        settings.parquet_path,
        settings.vectorstore_path,
        settings.whoosh_dir,
        settings.chunk_size,
        settings.chunk_overlap,
        None,
    )
    retriever = HybridRetriever(vs=vs, ix=ix, compressor=compressor, cross_encoder=cross_encoder)

    prompt = PromptTemplate.from_template(
        f"{settings.assistant_role}\n\n"
        "Context (up to 2,000 tokens, with document IDs):\n"
        "{context}\n"
        "Q: {question}\n"
        "A: Provide your answer."
    )

    app.state.rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    if not settings.skip_warmup:
        logger.info("Warming up RAG chain…")
        await app.state.rag_chain.ainvoke({"question": "warmup"})
        logger.info("RAG ready.")

    # --- 2) Dataframe Initialization --- #
    logger.info("Loading Parquet data from GCS…")
    fs = gcsfs.GCSFileSystem()
    with fs.open(settings.parquet_path, "rb") as f:
        df = pl.read_parquet(f)

    # lowercase for filtering
    for col in ("title", "status", "legalBasis"):
        df = df.with_columns(pl.col(col).str.to_lowercase().alias(f"_{col}_lc"))

    # materialize unique filter values
    app.state.df = df
    app.state.statuses     = df["_status_lc"].unique().to_list()
    app.state.legal_bases  = df["_legalBasis_lc"].unique().to_list()
    app.state.orgs_list    = df.explode("list_name")["list_name"].unique().to_list()
    app.state.countries_list = df.explode("list_country")["list_country"].unique().to_list()

    yield
    # teardown (if any) goes here

# ---------------------------------------------------------------------------- #
#                                   App Setup                                   #
# ---------------------------------------------------------------------------- #
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------- #
#                                 RAG Endpoint                                  #
# ---------------------------------------------------------------------------- #
class RAGRequest(BaseModel):
    session_id: Optional[str]
    query: str

class RAGResponse(BaseModel):
    answer: str
    source_ids: List[str]

def rag_chain_depender(app: FastAPI = Depends(lambda: app)) -> Any:
    chain = app.state.rag_chain
    if chain is None:
        raise HTTPException(status_code=500, detail="RAG chain not initialized")
    return chain

@app.post("/rag", response_model=RAGResponse)
async def ask_rag(
    req: RAGRequest,
    rag_chain = Depends(rag_chain_depender)
):
    result = await rag_chain.ainvoke({"question": req.query})
    sources = [doc.metadata.get("id", "") for doc in result.get("source_documents", [])]
    return RAGResponse(answer=result["answer"], source_ids=sources)

# ---------------------------------------------------------------------------- #
#                                  Data Endpoints                               #
# ---------------------------------------------------------------------------- #
@app.get("/api/projects")
def get_projects(
    page: int = 0,
    limit: int = 10,
    search: str = "",
    status: str = "",
    legalBasis: str = "",
    organization: str = "",
    country: str = "",
    fundingScheme: str = "",
    proj_id: str = "",
    sortOrder: str = "desc",
    sortField: str = "startDate",
):
    df: pl.DataFrame = app.state.df
    start = page * limit
    sel = df

    if search:
        sel = sel.filter(pl.col("_title_lc").str.contains(search.lower()))
    if status:
        sel = sel.filter(
            pl.col("status").is_null() if status == "UNKNOWN"
            else pl.col("_status_lc") == status.lower()
        )
    if legalBasis:
        sel = sel.filter(pl.col("_legalBasis_lc") == legalBasis.lower())
    if organization:
        sel = sel.filter(pl.col("list_name").list.contains(organization))
    if country:
        sel = sel.filter(pl.col("list_country").list.contains(country))
    if fundingScheme:
        sel = sel.filter(pl.col("fundingScheme").list.contains(fundingScheme))
    if proj_id:
        sel = sel.filter(pl.col("id") == proj_id)

    base_cols = [
        "id","title","status","startDate","endDate","ecMaxContribution","acronym",
        "legalBasis","objective","frameworkProgramme","list_euroSciVocTitle",
        "list_euroSciVocPath","totalCost","list_isPublishedAs","fundingScheme"
    ]
    # add shap/explanation columns
    for i in range(1,7):
        base_cols += [f"top{i}_feature", f"top{i}_shap"]
    base_cols += ["predicted_label","predicted_prob"]

    sort_desc = True if sortOrder=="desc" else False
    sortField = sortField if sortField in df.columns else "startDate"

    rows = (
        sel.sort(sortField, descending=sort_desc)
           .slice(start, limit)
           .select(base_cols)
           .to_dicts()
    )

    projects = []
    for row in rows:
        explanations = []
        for i in range(1,7):
            feat = row.pop(f"top{i}_feature", None)
            shap = row.pop(f"top{i}_shap", None)
            if feat is not None and shap is not None:
                explanations.append({"feature": feat, "shap": shap})
        row["explanations"] = explanations

        # publications aggregation
        raw_pubs = row.pop("list_publications", []) or []
        pub_counts: Dict[str,int] = {}
        for p in raw_pubs:
            pub_counts[p] = pub_counts.get(p, 0) + 1
        row["publications"] = pub_counts

        projects.append(row)

    return projects

@app.get("/api/filters")
def get_filters(request: Request):
    df = app.state.df
    params = request.query_params

    if s := params.get("status"):
        df = df.filter(pl.col("status").is_null() if s=="UNKNOWN"
                       else pl.col("_status_lc")==s.lower())
    if lb := params.get("legalBasis"):
        df = df.filter(pl.col("_legalBasis_lc")==lb.lower())
    if org := params.get("organization"):
        df = df.filter(pl.col("list_name").list.contains(org))
    if c := params.get("country"):
        df = df.filter(pl.col("list_country").list.contains(c))
    if search := params.get("search"):
        df = df.filter(pl.col("_title_lc").str.contains(search.lower()))

    def normalize(vals):
        return sorted({("UNKNOWN" if v is None else v) for v in vals})

    return {
        "statuses":     normalize(df["status"].to_list()),
        "legalBases":   normalize(df["legalBasis"].to_list()),
        "organizations": normalize(df["list_name"].explode().to_list()),
        "countries":    normalize(df["list_country"].explode().to_list()),
        "fundingSchemes": normalize(df["fundingScheme"].explode().to_list()),
        "ids":          normalize(df["id"].to_list()),
    }

@app.get("/api/stats")
def get_stats(request: Request):
    lf = app.state.df.lazy()
    params = request.query_params

    if s := params.get("status"):
        lf = lf.filter(pl.col("_status_lc")==s.lower())
    if lb := params.get("legalBasis"):
        lf = lf.filter(pl.col("_legalBasis_lc")==lb.lower())
    if org := params.get("organization"):
        lf = lf.filter(pl.col("list_name").list.contains(org))
    if c := params.get("country"):
        lf = lf.filter(pl.col("list_country").list.contains(c))
    if mn := params.get("minFunding"):
        lf = lf.filter(pl.col("ecMaxContribution")>=int(mn))
    if mx := params.get("maxFunding"):
        lf = lf.filter(pl.col("ecMaxContribution")<=int(mx))
    if y1 := params.get("minYear"):
        lf = lf.filter(pl.col("startDate").dt.year()>=int(y1))
    if y2 := params.get("maxYear"):
        lf = lf.filter(pl.col("startDate").dt.year()<=int(y2))

    grouped = (
        lf.select(pl.col("startDate").dt.year().alias("year"))
          .group_by("year")
          .agg(pl.count().alias("count"))
          .sort("year")
          .collect()
    )
    years, counts = grouped["year"].to_list(), grouped["count"].to_list()

    return {
        "Projects per Year":    {"labels": years, "values": counts},
        "Projects per Year 2":  {"labels": years, "values": counts},
        "Projects per Year 3":  {"labels": years, "values": counts},
        "Projects per Year 4":  {"labels": years, "values": counts},
        "Projects per Year 5":  {"labels": years, "values": counts},
        "Projects per Year 6":  {"labels": years, "values": counts},
    }

@app.get("/api/project/{project_id}/organizations")
def get_project_organizations(project_id: str):
    df = app.state.df
    sel = df.filter(pl.col("id")==project_id)
    if sel.is_empty():
        raise HTTPException(status_code=404, detail="Project not found")

    orgs_df = (
        sel.select([
            pl.col("list_name").explode().alias("name"),
            pl.col("list_city").explode().alias("city"),
            pl.col("list_SME").explode().alias("sme"),
            pl.col("list_role").explode().alias("role"),
            pl.col("list_organizationURL").explode().alias("orgURL"),
            pl.col("list_ecContribution").explode().alias("contribution"),
            pl.col("list_activityType").explode().alias("activityType"),
            pl.col("list_country").explode().alias("country"),
            pl.col("list_geolocation").explode().alias("geoloc"),
        ])
        .with_columns([
            pl.col("geoloc").str.split(",").alias("latlon"),
        ])
        .with_columns([
            pl.col("latlon").list.get(0).cast(pl.Float64).alias("latitude"),
            pl.col("latlon").list.get(1).cast(pl.Float64).alias("longitude"),
        ])
        .filter(pl.col("name").is_not_null())
        .select([
            "name","city","sme","role","contribution",
            "activityType","orgURL","country","latitude","longitude"
        ])
    )

    return orgs_df.to_dicts()
