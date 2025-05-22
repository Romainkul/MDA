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