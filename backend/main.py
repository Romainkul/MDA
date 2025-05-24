import logging
import os
import shutil
import tempfile
import traceback
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiofiles
import faiss
import gcsfs
import polars as pl
import torch
import zipfile
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, PrivateAttr
from pydantic_settings import BaseSettings as SettingsBase
from sentence_transformers import CrossEncoder
from starlette.concurrency import run_in_threadpool
from tqdm import tqdm
from transformers import (  # Transformers for LLM pipeline
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

# LangChain imports for RAG
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# Project-specific imports
from app.rag import build_indexes, HybridRetriever

# ---------------------------------------------------------------------------- #
#                                   Settings                                   #
# ---------------------------------------------------------------------------- #

class Settings(SettingsBase):
    """
    Configuration settings loaded from environment or .env file.
    """
    # Data sources
    parquet_path: str = "gs://mda_eu_project/data/consolidated_clean_pred.parquet"
    whoosh_dir: str = "gs://mda_eu_project/whoosh_index"
    vectorstore_path: str = "gs://mda_eu_project/vectorstore_index"

    # Model names
    embedding_model: str = "sentence-transformers/LaBSE"
    llm_model: str = "google/flan-t5-base"
    cross_encoder_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # RAG parameters
    chunk_size: int = 750
    chunk_overlap: int = 100
    hybrid_k: int = 2
    assistant_role: str = (
        "You are a knowledgeable project analyst. You have access to the following retrieved document snippets."
    )
    skip_warmup: bool = True

    # CORS
    allowed_origins: List[str] = ["*"]

    class Config:
        env_file = ".env"

# Instantiate settings and logger
settings = Settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pre-instantiate the embedding model for reuse
EMBEDDING = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
    model_kwargs={"trust_remote_code": True},
)

@lru_cache(maxsize=256)
def embed_query_cached(query: str) -> List[float]:
    """Cache embedding vectors for repeated queries."""
    return EMBEDDING.embed_query(query.strip().lower())

# ---------------------------------------------------------------------------- #
#                              Application Lifespan                            #
# ---------------------------------------------------------------------------- #

app = FastAPI(lifespan=lambda app: lifespan(app))

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: initialize RAG chain, embeddings, memory, indexes, and load data.
    Shutdown: clean up resources if needed.
    """
    # 1) Initialize document compressor
    logger.info("Initializing Document Compressor")
    compressor = DocumentCompressorPipeline(
        transformers=[EmbeddingsRedundantFilter(embeddings=EMBEDDING)]
    )

    # 2) Initialize and quantize Cross-Encoder
    logger.info("Initializing Cross-Encoder")
    cross_encoder = CrossEncoder(settings.cross_encoder_model)
    cross_encoder.model = torch.quantization.quantize_dynamic(
        cross_encoder.model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    logger.info("Cross-Encoder quantized")

    # 3) Build Seq2Seq pipeline and wrap in LangChain
    logger.info("Initializing LLM pipeline")
    tokenizer = T5Tokenizer.from_pretrained(settings.llm_model)
    model = T5ForConditionalGeneration.from_pretrained(settings.llm_model)
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    gen_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
    )
    llm = HuggingFacePipeline(pipeline=gen_pipe)

    # 4) Initialize conversation memory
    logger.info("Initializing Conversation Memory")
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
        k=5,
    )

    # 5) Build or load indexes for vectorstore and Whoosh
    logger.info("Building or loading indexes")
    vs, ix = await build_indexes(
        settings.parquet_path,
        settings.vectorstore_path,
        settings.whoosh_dir,
        settings.chunk_size,
        settings.chunk_overlap,
        None,
    )
    retriever = HybridRetriever(vs=vs, ix=ix, compressor=compressor, cross_encoder=cross_encoder)

    # 6) Define prompt template for RAG chain
    prompt = PromptTemplate.from_template(
        f"{settings.assistant_role}\n"  
        "{context}\n"
        "User Question:\n{question}\n"
        "Answer:"  # Rules are embedded in assistant_role
    )

    # 7) Instantiate the conversational retrieval chain
    logger.info("Initializing Retrieval Chain")
    app.state.rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    # Optional warmup
    if not settings.skip_warmup:
        logger.info("Warming up RAG chain")
        await app.state.rag_chain.ainvoke({"question": "warmup"})

    # 8) Load project data into Polars DataFrame
    logger.info("Loading Parquet data from GCS")
    fs = gcsfs.GCSFileSystem()
    with fs.open(settings.parquet_path, "rb") as f:
        df = pl.read_parquet(f)
    # Cast id to integer and lowercase key columns for filtering
    df = df.with_columns(
        pl.col("id").cast(pl.Int64),
        *(pl.col(col).str.to_lowercase().alias(f"_{col}_lc") for col in [
            "title", "status", "legalBasis", "fundingScheme"
        ])
    )

    # Cache DataFrame and filter values in app state
    app.state.df = df
    app.state.statuses = df["_status_lc"].unique().to_list()
    app.state.legal_bases = df["_legalBasis_lc"].unique().to_list()
    app.state.orgs_list = df.explode("list_name")["list_name"].unique().to_list()
    app.state.countries_list = df.explode("list_country")["list_country"].unique().to_list()

    yield  # Application is ready
    
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
#                                 Pydantic Models                             #
# ---------------------------------------------------------------------------- #

class RAGRequest(BaseModel):
    session_id: Optional[str] = None  # Optional conversation ID
    query: str  # User's query text

class RAGResponse(BaseModel):
    answer: str
    source_ids: List[str]

# ---------------------------------------------------------------------------- #
#                                 RAG Endpoint                                  #
# ---------------------------------------------------------------------------- #

def rag_chain_depender(app: FastAPI = Depends(lambda: app)) -> Any:
    """
    Dependency injector to retrieve the initialized RAG chain from the application state.
    Raises HTTPException if chain is not yet initialized.
    """
    chain = app.state.rag_chain
    if chain is None:
        # If the chain isn't set up, respond with a 500 server error
        raise HTTPException(status_code=500, detail="RAG chain not initialized")
    return chain

@app.post("/api/rag", response_model=RAGResponse)
async def ask_rag(
    req: RAGRequest,
    rag_chain = Depends(rag_chain_depender)
):
    """
    Endpoint to process a RAG-based query.

    1. Logs start of processing.
    2. Invokes the RAG chain asynchronously with the user question.
    3. Validates returned result structure and extracts answer + source IDs.
    4. Handles any exceptions by logging traceback and returning a JSON error.
    """
    logger.info("Starting to answer RAG query")
    try:
        # Asynchronously invoke the chain to get answer + docs
        result = await rag_chain.ainvoke({"question": req.query})
        logger.info("RAG results retrieved")

        # Validate that the chain returned expected dict
        if not isinstance(result, dict):
            # Try sync call for debugging
            result2 = await rag_chain.acall({"question": req.query})
            raise ValueError(
                f"Expected dict from chain, got {type(result)}; "
                f"acall() returned {type(result2)}"
            )

        # Extract answer text and source document IDs
        answer = result.get("answer")
        docs   = result.get("source_documents", [])
        sources = [d.metadata.get("id", "") for d in docs]

        return RAGResponse(answer=answer, source_ids=sources)

    except Exception as e:
        # Log full stacktrace to container logs
        traceback.print_exc()
        # Return HTTP 500 with error detail
        raise HTTPException(status_code=500, detail=str(e))
    

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
    """
    Paginated project listing with optional filtering and sorting.

    Query Parameters:
    - page: zero-based page index
    - limit: number of items per page
    - search: substring search in project title
    - status, legalBasis, organization, country, fundingScheme: filters
    - proj_id: exact project ID filter
    - sortOrder: 'asc' or 'desc'
    - sortField: field name to sort by (fallback to startDate)

    Returns a list of project dicts including explanations and publication counts.
    """
    df: pl.DataFrame = app.state.df
    start = page * limit
    sel = df

    # Apply text and field filters as needed
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
        sel = sel.filter(pl.col("_fundingScheme_lc").str.contains(fundingScheme.lower()))
    if proj_id:
        sel = sel.filter(pl.col("id") == int(proj_id))

    # Base columns to return
    base_cols = [
        "id","title","status","startDate","endDate","ecMaxContribution","acronym",
        "legalBasis","objective","frameworkProgramme","list_euroSciVocTitle",
        "list_euroSciVocPath","totalCost","list_isPublishedAs","fundingScheme"
    ]
    # Append top feature & SHAP value columns
    for i in range(1,7):
        base_cols += [f"top{i}_feature", f"top{i}_shap"]
    base_cols += ["predicted_label","predicted_prob"]

    # Determine sort direction and safe field
    sort_desc = sortOrder.lower() == "desc"
    sortField = sortField if sortField in df.columns else "startDate"

    # Query, sort, slice, and collect to Python dicts
    rows = (
        sel.sort(sortField, descending=sort_desc)
           .slice(start, limit)
           .select(base_cols)
           .to_dicts()
    )

    projects = []
    for row in rows:
        # Reformat SHAP explanations into list of dicts
        explanations = []
        for i in range(1,7):
            feat = row.pop(f"top{i}_feature", None)
            shap = row.pop(f"top{i}_shap", None)
            if feat is not None and shap is not None:
                explanations.append({"feature": feat, "shap": shap})
        row["explanations"] = explanations

        # Aggregate publications counts
        raw_pubs = row.pop("list_publications", []) or []
        pub_counts: Dict[str,int] = {}
        for p in raw_pubs:
            pub_counts[p] = pub_counts.get(p, 0) + 1
        row["publications"] = pub_counts

        projects.append(row)

    return projects

@app.get("/api/filters")
def get_filters(request: Request):
    """
    Retrieve available filter options based on current dataset and optional query filters.

    Returns JSON with lists for statuses, legalBases, organizations, countries, and fundingSchemes.
    """
    df = app.state.df
    params = request.query_params

    # Dynamically filter df based on provided params
    if s := params.get("status"):
        df = df.filter(pl.col("status").is_null() if s == "UNKNOWN"
                       else pl.col("_status_lc") == s.lower())
    if lb := params.get("legalBasis"):
        df = df.filter(pl.col("_legalBasis_lc") == lb.lower())
    if org := params.get("organization"):
        df = df.filter(pl.col("list_name").list.contains(org))
    if c := params.get("country"):
        df = df.filter(pl.col("list_country").list.contains(c))
    if search := params.get("search"):
        df = df.filter(pl.col("_title_lc").str.contains(search.lower()))

    def normalize(vals):
        # Map None to "UNKNOWN" and return sorted unique list
        return sorted({("UNKNOWN" if v is None else v) for v in vals})

    return {
        "statuses":     normalize(df["status"].to_list()),
        "legalBases":   normalize(df["legalBasis"].to_list()),
        "organizations": normalize(df["list_name"].explode().to_list())[:500],
        "countries":    normalize(df["list_country"].explode().to_list()),
        "fundingSchemes": normalize(df["fundingScheme"].explode().to_list()),
    }

@app.get("/api/stats")
def get_stats(request: Request):
    """
    Compute annual statistics on projects with optional filters for status, legal basis, etc.

    Returns a dict of chart data for projects per year.
    """
    lf = app.state.df.lazy()
    params = request.query_params

    # Apply lazy filters
    if s := params.get("status"):
        lf = lf.filter(pl.col("_status_lc") == s.lower())
    if lb := params.get("legalBasis"):
        lf = lf.filter(pl.col("_legalBasis_lc") == lb.lower())
    if org := params.get("organization"):
        lf = lf.filter(pl.col("list_name").list.contains(org))
    if c := params.get("country"):
        lf = lf.filter(pl.col("list_country").list.contains(c))
    if mn := params.get("minFunding"):
        lf = lf.filter(pl.col("ecMaxContribution") >= int(mn))
    if mx := params.get("maxFunding"):
        lf = lf.filter(pl.col("ecMaxContribution") <= int(mx))
    if y1 := params.get("minYear"):
        lf = lf.filter(pl.col("startDate").dt.year() >= int(y1))
    if y2 := params.get("maxYear"):
        lf = lf.filter(pl.col("startDate").dt.year() <= int(y2))

    # Group by year and count
    grouped = (
        lf.select(pl.col("startDate").dt.year().alias("year"))
          .group_by("year")
          .agg(pl.count().alias("count"))
          .sort("year")
          .collect()
    )
    years, counts = grouped["year"].to_list(), grouped["count"].to_list()

    # Return data ready for frontend charts
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
    """
    Retrieve organization details for a given project ID, including geolocation.

    Raises 404 if the project ID does not exist.
    """
    df = app.state.df
    sel = df.filter(pl.col("id") == int(project_id))
    if sel.is_empty():
        raise HTTPException(status_code=404, detail="Project not found")

    # Explode list columns and parse latitude/longitude
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
            # Split "lat,lon" string into list
            pl.col("geoloc").str.split(",").alias("latlon"),
        ])
        .with_columns([
            # Cast to floats for numeric use
            pl.col("latlon").list.get(0).cast(pl.Float64).alias("latitude"),
            pl.col("latlon").list.get(1).cast(pl.Float64).alias("longitude"),
        ])
        .filter(pl.col("name").is_not_null())
        .select([
            "name","city","sme","role","contribution",
            "activityType","orgURL","country","latitude","longitude"
        ])
    )
    logger.info(f"Organization data for project {project_id}: {orgs_df.to_dicts()}")
    return orgs_df.to_dicts()

