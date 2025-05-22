from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
# Access the global FastAPI app state
#from fastapi import current_app
from pydantic import BaseModel
#try:
#    from rag import get_rag_chain, RAGRequest, RAGResponse
#except:
#    from .rag import get_rag_chain, RAGRequest, RAGResponse
from contextlib import asynccontextmanager
import os
import polars as pl
import gcsfs

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Romain\OneDrive - KU Leuven\focal-pager-460414-e9-45369b738be0.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    bucket = "mda_eu_project"
    path   = "data/consolidated_clean_pred.parquet" #"data/consolidated_clean.parquet"
    uri    = f"gs://{bucket}/{path}"

    fs = gcsfs.GCSFileSystem()

    with fs.open(uri, "rb") as f:
        df = pl.read_parquet(f)

    for col in ("title", "status", "legalBasis"):
        df = df.with_columns(pl.col(col).str.to_lowercase().alias(f"_{col}_lc"))

    statuses      = df["_status_lc"].unique().to_list()
    legal_bases   = df["_legalBasis_lc"].unique().to_list()
    organizations = df.explode("list_name")["list_name"].unique().to_list()
    countries     = df.explode("list_country")["list_country"].unique().to_list()

    app.state.df             = df
    app.state.statuses       = statuses
    app.state.legal_bases    = legal_bases
    app.state.orgs_list      = organizations
    app.state.countries_list = countries

    yield
    
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        if status =="UNKNOWN":
            sel = sel.filter(pl.col("status").is_null())
        else:
            sel = sel.filter(pl.col("_status_lc") == status.lower())
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

    cols = [
        "id", "title", "status", "startDate", "endDate",
        "ecMaxContribution", "acronym", "legalBasis", "objective",
        "frameworkProgramme", "list_euroSciVocTitle", "list_euroSciVocPath","totalCost","list_isPublishedAs","fundingScheme"
    ]
    for i in range(1, 7):
        cols += [f"top{i}_feature", f"top{i}_shap"]

    cols += ["predicted_label", "predicted_prob"]
    sortOrder = True if sortOrder == "desc" else False
    sortField = sortField if sortField in df.columns else "startDate"
    rows = (
        sel.sort(sortField,descending=sortOrder).slice(start, limit)
           .select(cols)
           .to_dicts()
    )

    projects = []
    for row in rows:
        explanations = []
        for i in range(1, 7):
            feat = row.pop(f"top{i}_feature", None)
            shap = row.pop(f"top{i}_shap", None)
            if feat is not None and shap is not None:
                explanations.append({"feature": feat, "shap": shap})
        row["explanations"] = explanations
        # 2) transform list_publications into a { type: count } map
        raw_pubs = row.pop("list_publications", None) or []
        pub_counts: dict[str, int] = {}
        for entry in raw_pubs:
            # assuming entry is a string like "paper" or "peer reviewed paper"
            pub_counts[entry] = pub_counts.get(entry, 0) + 1

        row["publications"] = pub_counts

        projects.append(row)

    return projects


@app.get("/api/filters")
def get_filters(request: Request):
    df = app.state.df
    params = request.query_params

    # apply the same filters you use elsewhere
    if s := params.get("status"):
        if s == "UNKNOWN":
            df = df.filter(pl.col("status").is_null())
        else:
            df = df.filter(pl.col("_status_lc") == s.lower())
    if lb := params.get("legalBasis"):
        df = df.filter(pl.col("_legalBasis_lc") == lb.lower())
    if org := params.get("organization"):
        df = df.filter(pl.col("list_name").list.contains(org))
    if c := params.get("country"):
        df = df.filter(pl.col("list_country").list.contains(c))
    if search := params.get("search"):
        df = df.filter(pl.col("_title_lc").str.contains(search.lower()))

    def normalize(values):
        return sorted(set("UNKNOWN" if v is None else v for v in values))

    return {
        "statuses": normalize(df["status"].to_list()),
        "legalBases": normalize(df["legalBasis"].to_list()),
        "organizations": normalize(df["list_name"].explode().to_list()),
        "countries": normalize(df["list_country"].explode().to_list()),
        "fundingSchemes": normalize(df["fundingScheme"].explode().to_list()),
        "ids": normalize(df["id"].to_list()),
    }


@app.get("/api/stats")
def get_stats(request: Request):
    params = request.query_params
    lf = app.state.df.lazy()

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

    grouped = (
        lf.select(pl.col("startDate").dt.year().alias("year"))
          .group_by("year")
          .agg(pl.count().alias("count"))
          .sort("year")
          .collect()
    )
    years, counts = grouped["year"].to_list(), grouped["count"].to_list()
    return {"Projects per Year": {"labels": years, "values": counts},
            "Projects per Year 2": {"labels": years, "values": counts},
             "Projects per Year 3": {"labels": years, "values": counts},
              "Projects per Year 4": {"labels": years, "values": counts},
               "Projects per Year 5": {"labels": years, "values": counts},
                "Projects per Year 6": {"labels": years, "values": counts}}


@app.get("/api/project/{project_id}/organizations")
def get_project_organizations(project_id: str):
    df = app.state.df

    sel = df.filter(pl.col("id") == project_id)
    if sel.is_empty():
        raise HTTPException(status_code=404, detail="Project not found")

    orgs_df = (
        sel
        .select([
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
            # now this is a List(Utf8)
            pl.col("geoloc").str.split(",").alias("latlon"),
        ])
        .with_columns([
            pl.col("latlon").list.get(0).cast(pl.Float64).alias("latitude"),
            pl.col("latlon").list.get(1).cast(pl.Float64).alias("longitude"),
        ])
        .filter(pl.col("name").is_not_null())
        .select(["name", "city", "sme","role","contribution","activityType","orgURL","country", "latitude", "longitude"])
    )

    return orgs_df.to_dicts()

"""def rag_chain_depender():
    """
#Dependency injector for the RAG chain stored in app.state.
#Raises HTTPException if not initialized.
"""
    chain = current_app.state.rag_chain
    if chain is None:
        raise HTTPException(status_code=500, detail="RAG chain not initialized")
    return chain

@app.post("/rag", response_model=RAGResponse)
async def ask_rag(
    req: RAGRequest,
    rag_chain = Depends(rag_chain_depender)
):
    """
#Handle a RAG query. Uses session memory and the provided RAG chain.
"""
    # Invoke the chain with the named input
    result = await rag_chain.ainvoke({"question": req.query})
    return RAGResponse(answer=result["answer"])"""