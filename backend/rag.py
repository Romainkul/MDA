import os
import logging
from typing import Any, Dict, List, Optional, Tuple, AsyncGenerator
from contextlib import asynccontextmanager

import gcsfs
import aiofiles
import polars as pl
from pydantic_settings import BaseSettings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

from transformers import AutoTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
#from optimum.onnxruntime import ORTModelForCausalLM, ORTOptimizer
from sentence_transformers import CrossEncoder
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import MultifieldParser
from tqdm import tqdm
import faiss

from functools import lru_cache

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

# === Global Embeddings & Cache ===
EMBEDDING = HuggingFaceEmbeddings(model_name=settings.embedding_model)

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
