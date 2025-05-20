from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import polars as pl
import gcsfs

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ——— Load once, in memory ———
    bucket = "mda_eu_project"
    path   = "data/consolidated_clean.parquet"
    uri    = f"gs://{bucket}/{path}"

    # gcsfs will pick up GOOGLE_APPLICATION_CREDENTIALS automatically
    fs = gcsfs.GCSFileSystem()

    # Open a streaming handle & let Polars read it directly
    with fs.open(uri, "rb") as f:
        df = pl.read_parquet(f)

    # Pre-normalize lowercase text columns for fast filtering
    for col in ("title", "status", "legalBasis"):
        df = df.with_columns(pl.col(col).str.to_lowercase().alias(f"_{col}_lc"))

    # Cache filter dropdowns
    statuses      = df["_status_lc"].unique().to_list()
    legal_bases   = df["_legalBasis_lc"].unique().to_list()
    organizations = df.explode("list_name")["list_name"].unique().to_list()
    countries     = df.explode("list_country")["list_country"].unique().to_list()

    # Store in app.state
    app.state.df             = df
    app.state.statuses       = statuses
    app.state.legal_bases    = legal_bases
    app.state.orgs_list      = organizations
    app.state.countries_list = countries

    yield
    # (no teardown needed)

# Create app with our lifespan
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/projects")
def get_projects(page: int = 0, limit: int = 10, search: str = "", status: str = ""):
    df = app.state.df
    start = page * limit
    sel = df

    if search:
        sel = sel.filter(pl.col("_title_lc").str.contains(search.lower()))

    if status:
        sel = sel.filter(pl.col("_status_lc") == status.lower())

    return (
        sel.slice(start, limit)
           .select([
               "id","title","status","startDate","ecMaxContribution",
               "acronym","endDate","legalBasis","objective",
               "frameworkProgramme","list_euroSciVocTitle",
               "list_euroSciVocPath"
           ])
           .to_dicts()
    )


@app.get("/api/filters")
def get_filters():
    return {
        "statuses":      app.state.statuses,
        "legalBases":    app.state.legal_bases,
        "organizations": app.state.orgs_list,
        "countries":     app.state.countries_list
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
          .groupby("year")
          .agg(pl.count().alias("count"))
          .sort("year")
          .collect()
    )
    years, counts = grouped["year"].to_list(), grouped["count"].to_list()
    return {"Projects per Year": {"labels": years, "values": counts}}


@app.get("/api/project/{project_id}/organizations")
def get_project_organizations(project_id: str):
    df = app.state.df
    sel = df.filter(pl.col("id") == project_id)
    if sel.is_empty():
        raise HTTPException(status_code=404, detail="Project not found")

    orgs_df = (
        sel.select([
            pl.explode("list_name").alias("name"),
            pl.explode("list_country").alias("country"),
            pl.explode("list_geolocation").alias("geoloc"),
        ])
        .with_columns([
            pl.col("geoloc")
              .str.split_exact(",", 1)
              .alias("latlon")
        ])
        .with_columns([
            pl.col("latlon").list.get(0).cast(float).alias("latitude"),
            pl.col("latlon").list.get(1).cast(float).alias("longitude")
        ])
        .filter(pl.col("name").is_not_null())
        .select(["name","country","latitude","longitude"])
    )

    return orgs_df.to_dicts()
