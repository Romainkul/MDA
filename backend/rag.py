import os
import io
import asyncio
from functools import lru_cache
from typing import Any, Dict, List

import gcsfs
import polars as pl
from pydantic_settings import BaseSettings
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel

from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import CrossEncoder

from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import MultifieldParser

# === Settings ===
class Settings(BaseSettings):
    # allow GCS paths (e.g. "gs://bucket/...") via gcsfs
    parquet_path: str = "/content/drive/MyDrive/consolidated_clean.parquet"
    vectorstore_path: str = "/content/drive/MyDrive/vectorstore_index"
    whoosh_dir: str = "whoosh_index"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    llm_model: str = "bigscience/bloomz-560m"
    cross_encoder_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    chunk_size: int = 300
    chunk_overlap: int = 50
    hybrid_k: int = 50
    assistant_role: str = "You are a concise, factual assistant. Cite Document [ID] for each claim."

    class Config:
        env_file = ".env"

settings = Settings()

# === Globals ===
embedding: HuggingFaceEmbeddings  # will be set in lifespan
cross_encoder: CrossEncoder
compressor: DocumentCompressorPipeline
session_meta: Dict[str, Dict] = {}  # per‐session metadata
# rag_chain will be stored on app.state

# === Whoosh Index Builder ===
_WHOOSH_CACHE: Dict[str, index.Index] = {}
def build_whoosh_index(docs: List[Document]) -> index.Index:
    key = settings.whoosh_dir
    if key in _WHOOSH_CACHE:
        return _WHOOSH_CACHE[key]
    os.makedirs(key, exist_ok=True)
    schema = Schema(id=ID(stored=True, unique=True),
                    content=TEXT(analyzer=StemmingAnalyzer()))
    ix = index.create_in(key, schema)
    writer = ix.writer()
    for doc in docs:
        writer.add_document(id=doc.metadata["id"], content=doc.page_content)
    writer.commit()
    _WHOOSH_CACHE[key] = ix
    return ix

# === Load Documents ===
def load_documents(path: str) -> List[Document]:
    # support GCS via gcsfs
    if path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem()
        with fs.open(path, "rb") as f:
            df = pl.read_parquet(f).to_pandas()
    else:
        df = pl.read_parquet(path).to_pandas()
    df = df.head(20)
    docs = []
    for _, row in df.iterrows():
        text = f"{row.get('title','')} {row.get('objective','')} {row.get('list_description','')}"
        meta = {c: str(row[c]) for c in row.index 
                if c in ("id","startDate","topics","list_euroSciVocTitle","list_euroSciVocPath")}
        docs.append(Document(page_content=text, metadata=meta))
    return docs

# === Query Expansion & BM25 ===
def expand_query_with_metadata(query: str, md: Dict[str,str]) -> str:
    parts = [query]
    for key,label in [
        ("startDate","from"), ("topics","about"),
        ("list_euroSciVocTitle","with topics"), ("list_euroSciVocPath","with topic path")
    ]:
        if md.get(key):
            parts.append(f"{label} {md[key]}")
    return "; ".join(parts)

async def iterative_retrieval(
    query: str, vs: FAISS, ix: index.Index,
    rewriter: HuggingFacePipeline, md: Dict[str,str]
) -> List[Document]:
    first_q = await rewriter.ainvoke({"query": query})
    parser = MultifieldParser(["content"], schema=ix.schema)
    def bm25(q: str) -> List[Document]:
        with ix.searcher() as s:
            hits = s.search(parser.parse(q), limit=settings.hybrid_k)
            return [
                Document(page_content=h["content"], metadata={"id": h["id"]})
                for h in hits
            ]
    docs1 = bm25(first_q)
    docs2 = bm25(expand_query_with_metadata(first_q, md))
    uniq = {d.metadata["id"]: d for d in docs1 + docs2}
    return list(uniq.values())

# === Cached Embeddings ===
@lru_cache(maxsize=1024)
def embed_query_cached(q: str):
    return embedding.embed_query(q)

# === Retriever & Index Builder ===
def build_retriever(docs: List[Document], embedder: HuggingFaceEmbeddings):
    ix = build_whoosh_index(docs)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    if os.path.exists(settings.vectorstore_path):
        vs = FAISS.load_local(
            settings.vectorstore_path,
            embeddings=embedder,
            allow_dangerous_deserialization=True
        )
    else:
        vs = FAISS.from_documents(chunks, embedder)
        vs.save_local(settings.vectorstore_path)
    return vs, ix

# === Rewriter & LLM Pipelines ===
rewriter_pipe = pipeline(
    "text-generation",
    model=AutoModelForCausalLM.from_pretrained(settings.llm_model),
    tokenizer=AutoTokenizer.from_pretrained(settings.llm_model),
    max_new_tokens=64, truncation=True, do_sample=False, return_full_text=False
)
rewriter = HuggingFacePipeline(pipeline=rewriter_pipe)
query_rewriter = PromptTemplate.from_template("Rewrite question: {query}") | rewriter

def get_llm_pipeline():
    gen_pipe = pipeline(
        "text-generation",
        model=AutoModelForCausalLM.from_pretrained(settings.llm_model),
        tokenizer=AutoTokenizer.from_pretrained(settings.llm_model),
        max_new_tokens=256, truncation=True,
        temperature=0.7, do_sample=True, return_full_text=False
    )
    return HuggingFacePipeline(pipeline=gen_pipe)

combine_prompt = PromptTemplate.from_template(
    f"""{settings.assistant_role}

Context (up to 2,000 tokens):
{{context}}

Q: {{question}}
A:
"""
)

async def retrieve_and_rank(
    query: str, vs: FAISS, ix: index.Index, md: Dict[str,str]
) -> List[Document]:
    bm25_docs = await iterative_retrieval(query, vs, ix, query_rewriter, md)
    dense = vs.similarity_search_by_vector(embed_query_cached(query), k=settings.hybrid_k)
    candidates = bm25_docs + dense
    scores = cross_encoder.predict([(query, d.page_content) for d in candidates])
    top_docs = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)[:settings.hybrid_k]]
    return compressor.compress_documents(top_docs, query=query)

class HybridRetriever(BaseRetriever):
    vs: Any
    ix: Any
    meta_store: Dict[str,Dict]

    def __init__(self, vs, ix, meta_store):
        super().__init__(vs=vs, ix=ix, meta_store=meta_store)
    def _get_relevant_documents(self, query: str):
        return asyncio.get_event_loop().run_until_complete(
            retrieve_and_rank(query, self.vs, self.ix, self.meta_store.get(query, {}))
        )
    async def _aget_relevant_documents(self, query: str):
        return await retrieve_and_rank(query, self.vs, self.ix, self.meta_store.get(query, {}))

# === FastAPI Lifespan & Initialization ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding, cross_encoder, compressor

    # 1) Embeddings & docs
    embedding = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    docs = load_documents(settings.parquet_path)

    # 2) Retriever + Whoosh index
    vs, ix = build_retriever(docs, embedding)

    # 3) Compressor and reranker
    compressor = DocumentCompressorPipeline(
        transformers=[EmbeddingsRedundantFilter(embeddings=embedding)]
    )
    cross_encoder = CrossEncoder(settings.cross_encoder_model)

    # 4) LLM & memory
    llm = get_llm_pipeline()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )

    # 5) Build RAG chain
    retriever = HybridRetriever(vs, ix, session_meta)
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": combine_prompt},
        return_source_documents=True
    )

    # 6) Warm-up
    vs.similarity_search("warmup", k=1)
    embed_query_cached("warmup")
    await rag_chain.ainvoke({"question": "warmup"})

    # 7) Store on app.state so get_rag_chain() can find it
    app.state.rag_chain = rag_chain

    yield

# === Pydantic Models & FastAPI Setup ===
class RAGRequest(BaseModel):
    session_id: str
    topic: str
    startDate: str
    query: str

class RAGResponse(BaseModel):
    answer: str