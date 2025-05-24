# Standard library
import logging
import os
import shutil
import tempfile
import traceback
import zipfile
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

# Third-party
import aiofiles
import faiss
import gcsfs
import polars as pl
import pickle
import torch
from tqdm import tqdm

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, PrivateAttr
from pydantic_settings import BaseSettings
from sentence_transformers import CrossEncoder
from starlette.concurrency import run_in_threadpool
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline,
)

from whoosh import index
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, Schema, TEXT
from whoosh.qparser import MultifieldParser

# LangChain
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings


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
    llm_model:           str = "google/flan-t5-base"
    cross_encoder_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # RAG parameters
    chunk_size:    int = 750
    chunk_overlap: int = 100
    hybrid_k:      int = 2
    assistant_role: str = (
        "You are a knowledgeable project analyst.  You have access to the following retrieved document snippets."
    )
    skip_warmup: bool = True
    allowed_origins: List[str] = ["*"]

    class Config:
        env_file = ".env"

settings = Settings()

# === Global Embeddings & Cache ===
EMBEDDING = HuggingFaceEmbeddings(model_name=settings.embedding_model)

@lru_cache(maxsize=128)
def embed_query_cached(query: str) -> List[float]:
    """Cache embedding vectors for queries."""
    return EMBEDDING.embed_query(query.strip().lower())

# === Whoosh Cache & Builder ===
async def build_whoosh_index(docs: List[Document], whoosh_dir: str) -> index.Index:
    """
    If gs://.../whoosh_index.zip exists, download & extract it once.
    Otherwise build locally from docs and upload the ZIP back to GCS.
    """
    fs = gcsfs.GCSFileSystem()
    is_gcs = whoosh_dir.startswith("gs://")
    zip_uri = whoosh_dir.rstrip("/") + ".zip"

    local_zip = "/tmp/whoosh_index.zip"
    local_dir = "/tmp/whoosh_index"

    # Clean slate
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # 1️⃣ Try downloading the ZIP if it exists on GCS
    if is_gcs and await run_in_threadpool(fs.exists, zip_uri):
        logger.info("Found whoosh_index.zip on GCS; downloading…")
        await run_in_threadpool(fs.get, zip_uri, local_zip)
        # Extract all files (flat) into local_dir
        with zipfile.ZipFile(local_zip, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                filename = os.path.basename(member.filename)
                if not filename:
                    continue
                target = os.path.join(local_dir, filename)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
        logger.info("Whoosh index extracted from ZIP.")
    else:
        logger.info("No whoosh_index.zip found; building index from docs.")

        # Define the schema with stored content
        schema = Schema(
            id=ID(stored=True, unique=True),
            content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        )

        # Create the index
        ix = index.create_in(local_dir, schema)
        writer = ix.writer()
        for doc in docs:
            writer.add_document(
                id=doc.metadata.get("id", ""),
                content=doc.page_content,
            )
        writer.commit()
        logger.info("Whoosh index built locally.")

        # Upload the ZIP back to GCS
        if is_gcs:
            logger.info("Zipping and uploading new whoosh_index.zip to GCS…")
            with zipfile.ZipFile(local_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(local_dir):
                    for fname in files:
                        full = os.path.join(root, fname)
                        arc = os.path.relpath(full, local_dir)
                        zf.write(full, arc)
            await run_in_threadpool(fs.put, local_zip, zip_uri)
            logger.info("Uploaded whoosh_index.zip to GCS.")

    # 2️⃣ Finally open the index and return it
    ix = index.open_dir(local_dir)
    return ix

# === Document Loader ===
async def load_documents(
    path: str,
    sample_size: Optional[int] = None
) -> List[Document]:
    """
    Load project data from a Parquet file (local path or GCS URI),
    assemble metadata context for each row, and return as Document objects.
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
    docs: List[Document],
    vectorstore_path: str,
    batch_size: int = 15000
) -> FAISS:
    """
    Expects a ZIP at vectorstore_path + ".zip" containing:
      - index.faiss
      - index.pkl
    Files may be nested under a subfolder (e.g. vectorstore_index_colab/).
    If the ZIP exists on GCS, download & load only.
    Otherwise, build from `docs`, save, re-zip, and upload.
    """
    fs = gcsfs.GCSFileSystem()
    is_gcs = vectorstore_path.startswith("gs://")
    zip_uri = vectorstore_path.rstrip("/") + ".zip"

    local_zip = "/tmp/faiss_index.zip"
    local_dir = "/tmp/faiss_store"

    # 1) if ZIP exists, download & extract
    if is_gcs and await run_in_threadpool(fs.exists, zip_uri):
        logger.info("Found FAISS ZIP on GCS; loading only.")
        # clean slate
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        os.makedirs(local_dir, exist_ok=True)

        # download zip
        await run_in_threadpool(fs.get, zip_uri, local_zip)

        # extract
        def _extract():
            with zipfile.ZipFile(local_zip, "r") as zf:
                zf.extractall(local_dir)
        await run_in_threadpool(_extract)

        # locate the two files anywhere under local_dir
        idx_path = None
        meta_path = None
        for root, _, files in os.walk(local_dir):
            if "index.faiss" in files:
                idx_path = os.path.join(root, "index.faiss")
            if "index.pkl" in files:
                meta_path = os.path.join(root, "index.pkl")
        if not idx_path or not meta_path:
            raise FileNotFoundError("Couldn't find index.faiss or index.pkl in extracted ZIP.")

        # memory-map load
        mmap_index = await run_in_threadpool(
            faiss.read_index, idx_path, faiss.IO_FLAG_MMAP
        )

        # load metadata
        with open(meta_path, "rb") as f:
            saved = pickle.load(f)

        # unpack metadata
        if isinstance(saved, tuple):
            _, docstore, index_to_docstore = (
                saved if len(saved) == 3 else (None, *saved)
            )
        else:
            docstore = getattr(saved, "docstore", saved._docstore)
            index_to_docstore = getattr(
                saved,
                "index_to_docstore",
                getattr(saved, "_index_to_docstore", saved._faiss_index_to_docstore)
            )

        # reconstruct FAISS
        vs = FAISS(
            embedding_function=EMBEDDING,
            index=mmap_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore,
        )
        logger.info("FAISS index loaded from ZIP.")
        return vs

    # 2) otherwise, build from scratch and upload
    logger.info("No FAISS ZIP found; building index from scratch.")
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    vs: FAISS = None
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        if vs is None:
            vs = FAISS.from_documents(batch, EMBEDDING)
        else:
            vs.add_documents(batch)
    assert vs is not None, "No documents to index!"

    # save locally
    vs.save_local(local_dir)

    if is_gcs:
        # re-zip all contents of local_dir (flattened)
        def _zip_dir():
            with zipfile.ZipFile(local_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(local_dir):
                    for fname in files:
                        full = os.path.join(root, fname)
                        arc = os.path.relpath(full, local_dir)
                        zf.write(full, arc)
        await run_in_threadpool(_zip_dir)
        await run_in_threadpool(fs.put, local_zip, zip_uri)
        logger.info("Built FAISS index and uploaded ZIP to GCS.")

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
    """
    Load documents, build/load Whoosh and FAISS indices, and return both.
    """
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