import os
import asyncio
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseSettings, BaseModel

from langchain.schema import Document, BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import MultifieldParser

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import CrossEncoder

# === Settings ===

class Settings(BaseSettings):
    parquet_path: str
    vectorstore_path: str
    whoosh_dir: str
    embedding_model: str
    llm_model: str
    cross_encoder_model: str
    chunk_size: int = 300
    chunk_overlap: int = 50
    hybrid_k: int = 50
    class Config:
        env_file = ".env"

settings = Settings()

# === Cache Whoosh index ===

_WHOOSH_CACHE: Dict[str, index.Index] = {}

def build_whoosh_index(docs: List[Document]) -> index.Index:
    """
    Builds or loads a cached Whoosh index for BM25 retrieval.
    """
    key = settings.whoosh_dir
    if key in _WHOOSH_CACHE:
        return _WHOOSH_CACHE[key]
    os.makedirs(settings.whoosh_dir, exist_ok=True)
    schema = Schema(id=ID(stored=True, unique=True), content=TEXT(analyzer=StemmingAnalyzer()))
    ix = index.create_in(settings.whoosh_dir, schema)
    writer = ix.writer()
    for doc in docs:
        writer.add_document(id=doc.metadata['id'], content=doc.page_content)
    writer.commit()
    _WHOOSH_CACHE[key] = ix
    return ix

# === Document loading ===

def load_documents(path: str) -> List[Document]:
    """
    Loads documents from a Parquet file and returns LangChain Documents.
    """
    import pandas as pd
    df = pd.read_parquet(path)
    docs = []
    for _, row in df.iterrows():
        text = f"{row['title']} {row['objective']} {row['list_description']}"
        meta = {c: str(row[c]) for c in row.index if c in ['id','startDate','topics']}
        docs.append(Document(page_content=text, metadata=meta))
    return docs

# === Retriever & Index builder ===

def build_retriever(docs: List[Document], embedding: HuggingFaceEmbeddings) -> Tuple[FAISS, index.Index]:
    """
    Builds or loads a FAISS index (sharded) and the Whoosh index.
    """
    ix = build_whoosh_index(docs)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    if os.path.exists(settings.vectorstore_path):
        vs = FAISS.load_local(
            settings.vectorstore_path,
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )
    else:
        vs = FAISS.from_documents(chunks, embedding)
        vs.save_local(settings.vectorstore_path)
    return vs, ix

# === Retrieval pipeline ===

def expand_query_with_metadata(query: str, metadata: Dict[str, str]) -> str:
    """
    Appends metadata filters to the query string.
    """
    parts = [query]
    if metadata.get('startDate'):
        parts.append(f"from {metadata['startDate']}")
    if metadata.get('topics'):
        parts.append(f"about {metadata['topics']}")
    return "; ".join(parts)

async def iterative_retrieval(
    query: str,
    vs: FAISS,
    ix: index.Index,
    rewriter: HuggingFacePipeline,
    metadata: Dict[str, str]
) -> List[Document]:
    """
    Performs BM25 searches on original and expanded queries.
    """
    first_q = await rewriter.ainvoke(query)
    parser = MultifieldParser(["content"], schema=ix.schema)
    def bm25_search(q: str) -> List[Document]:
        with ix.searcher() as s:
            hits = s.search(parser.parse(q), limit=settings.hybrid_k)
            return [Document(page_content=h['content'], metadata={'id':h['id']}) for h in hits]
    docs1 = bm25_search(first_q)
    docs2 = bm25_search(expand_query_with_metadata(first_q, metadata))
    unique = {d.metadata['id']: d for d in docs1 + docs2}
    return list(unique.values())

async def retrieve_and_rank(
    query: str,
    vs: FAISS,
    ix: index.Index,
    metadata: Dict[str, str]
) -> List[Document]:
    """
    Combines iterative BM25, dense retrieval, cross-encoding, and compression.
    """
    reranked = await iterative_retrieval(query, vs, ix, query_rewriter, metadata)
    dense = vs.similarity_search_by_vector(embed_query_cached(query), k=settings.hybrid_k)
    candidates = reranked + dense
    scores = cross_encoder.predict([(query, d.page_content) for d in candidates])
    top = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)[:settings.hybrid_k]]
    return compressor.compress_documents(top)

# === Hybrid Retriever ===

class HybridRetriever(BaseRetriever):
    """
    Wraps the retrieval pipeline as a LangChain BaseRetriever.
    """
    vs: Any
    ix: Any
    meta_store: Dict[str, Dict]
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, vs: FAISS, ix: index.Index, meta_store: Dict[str, Dict]):
        super().__init__(vs=vs, ix=ix, meta_store=meta_store)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return asyncio.get_event_loop().run_until_complete(
            retrieve_and_rank(query, self.vs, self.ix, self.meta_store.get(query, {}))
        )

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return await retrieve_and_rank(query, self.vs, self.ix, self.meta_store.get(query, {}))

# === Initialization & Lifespan ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: initialize embeddings, indexes, pipelines, and RAG chain.
    """
    global compressor, cross_encoder, query_rewriter, embed_query_cached, rag_chain
    # Embeddings & indexes
    embedding = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    docs      = load_documents(settings.parquet_path)
    vs, ix    = build_retriever(docs, embedding)
    compressor = DocumentCompressorPipeline([
        EmbeddingsRedundantFilter(embeddings=embedding)
    ])
    cross_encoder = CrossEncoder(settings.cross_encoder_model)

    # Query rewriter
    pipe = pipeline(
        "text-generation",
        model=AutoModelForCausalLM.from_pretrained(settings.llm_model),
        tokenizer=AutoTokenizer.from_pretrained(settings.llm_model),
        max_length=64, truncation=True, do_sample=False
    )
    query_rewriter = PromptTemplate.from_template("Rewrite question: {query}") | HuggingFacePipeline(pipeline=pipe)

    # LLM for answer generation
    gen_pipe = pipeline(
        "text-generation",
        model=AutoModelForCausalLM.from_pretrained(settings.llm_model),
        tokenizer=AutoTokenizer.from_pretrained(settings.llm_model),
        max_length=1024, truncation=True, temperature=0.7, do_sample=True
    )
    llm = HuggingFacePipeline(pipeline=gen_pipe)

    # Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )

    # Build hybrid retriever and RAG chain
    retriever = HybridRetriever(vs, ix, {})
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(ASSISTANT_ROLE)},
        return_source_documents=True
    )

    # Warm-up
    await asyncio.to_thread(vs.similarity_search, "warmup", 1)
    await asyncio.to_thread(embed_query_cached, "warmup")
    await rag_chain.ainvoke({"question": "warmup"})

    app.state.rag_chain = rag_chain
    yield

# Attach lifespan to app
app = FastAPI(lifespan=lifespan)

# DI helper
def get_rag_chain() -> ConversationalRetrievalChain:
    """
    Returns the initialized RAG chain from app.state.
    """
    from fastapi import current_app
    chain = current_app.state.rag_chain
    if chain is None:
        raise RuntimeError("RAG chain not initialized")
    return chain

# Pydantic models
class RAGRequest(BaseModel):
    session_id: str
    topic:      str
    startDate:  str
    query:      str

class RAGResponse(BaseModel):
    answer: str