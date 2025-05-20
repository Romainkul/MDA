from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import polars as pl
import pandas as pd
import datetime
from pathlib import Path
import os
import polars as pl
import gcsfs
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    bucket = "mda_eu_project"
    path_in_bucket = "data/consolidated_clean.parquet"
    gcs_path = f"gs://{bucket}/{path_in_bucket}"

    fs = gcsfs.GCSFileSystem()  
    app.state.df = pl.read_parquet(
        gcs_path,
        storage_options={"gcs": fs}
    )

    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/projects")
def get_projects(page: int = 0, limit: int = 10, search: str = "", status: str = ""):
    start = page * limit
    df_filt = app.state.df

    if search:
        df_filt = df_filt.filter(pl.col("title").str.to_lowercase().str.contains(search.lower()))

    if status:
        df_filt = df_filt.filter(pl.col("status").str.to_lowercase().str.contains(status.lower()))

    subset = df_filt[start:start + limit]
    return subset.select([
        "id",
        "title",
        "status",
        "startDate",
        "ecMaxContribution", "acronym","endDate","legalBasis","objective","frameworkProgramme","list_euroSciVocTitle","list_euroSciVocPath"
    ]).to_dicts()

@app.get("/api/filters")
def get_filters(status: str = "", organization: str = "", country: str = "", legalBasis: str = ""):
    dff = app.state.df.lazy()

    if status:
        dff = dff.filter(pl.col("status").str.to_lowercase() == status.lower())

    if organization:
        dff = dff.filter(pl.col("list_name").list.contains(organization))

    if country:
        dff = dff.filter(pl.col("list_country").list.contains(country))

    if legalBasis:
        dff = dff.filter(pl.col("legalBasis").str.to_lowercase() == legalBasis.lower())

    def extract_unique(col_name):
        return sorted(set(
            x for row in dff.select(col_name).drop_nulls().to_series().to_list()
            if isinstance(row, list) for x in row if x is not None
        ))

    return {
        "statuses": dff.select("status").drop_nulls().unique().to_series().to_list(),
        "organizations": extract_unique("list_name"),
        "countries": extract_unique("list_country"),
        "legalBases": dff.select("legalBasis").drop_nulls().unique().to_series().to_list()
    }

@app.get("/api/stats")
def get_stats(request: Request):
    query = dict(request.query_params)
    dff = app.state.df.lazy()

    # String filters (case-insensitive)
    if status := query.get("status"):
        dff = dff.filter(pl.col("status").str.to_lowercase() == status.lower())

    if org := query.get("organization"):
        dff = dff.filter(pl.col("list_name").list.contains(org))

    if country := query.get("country"):
        dff = dff.filter(pl.col("list_country").list.contains(country))

    if legal := query.get("legalBasis"):
        dff = dff.filter(pl.col("legalBasis").str.to_lowercase() == legal.lower())

    # Numeric range: funding
    if min_funding := query.get("minFunding"):
        dff = dff.filter(pl.col("ecMaxContribution") >= int(min_funding))

    if max_funding := query.get("maxFunding"):
        dff = dff.filter(pl.col("ecMaxContribution") <= int(max_funding))

    # Date range: startDate
    if min_year := query.get("minYear"):
        dff = dff.filter(pl.col("startDate").dt.year() >= int(min_year))

    if max_year := query.get("maxYear"):
        dff = dff.filter(pl.col("startDate").dt.year() <= int(max_year))

    grouped = (
        dff.select(pl.col("startDate").dt.year().alias("year"))
           .group_by("year")
           .agg(pl.count().alias("count"))
           .sort("year")
           .collect(streaming=True)
    )

    years = grouped["year"].to_list()
    counts = grouped["count"].to_list()

    return {
        "Projects per Year": {
            "labels": years,
            "values": counts
        },
        "Projects per Year 2": {
            "labels": years,
            "values": counts
        },
        "Projects per Year 3": {
            "labels": years,
            "values": counts
        },
        "Projects per Year 4": {
            "labels": years,
            "values": counts
        },
        "Projects per Year 5": {
            "labels": years,
            "values": counts
        },
        "Projects per Year 6": {
            "labels": years,
            "values": counts
        }
    }

@app.get("/api/project/{project_id}/organizations")
def get_project_organizations(project_id: str):
    try:
        dff = app.state.df.filter(pl.col("id") == project_id)

        if dff.height == 0:
            raise HTTPException(status_code=404, detail="Project not found")

        orgs = []
        for row in dff.to_dicts():
            names   = row.get("list_name", [])
            countries = row.get("list_country", [])
            geos    = row.get("list_geolocation", [])
            for name, country, latlon in zip(names, countries, geos):
                if name and latlon:
                    try:
                        lat_str, lon_str = latlon.split(",")
                        lat = float(lat_str.strip())
                        lon = float(lon_str.strip())

                        orgs.append({
                            "name": name,
                            "country": country,
                            "latitude": lat,
                            "longitude": lon,
                        })
                    except ValueError:
                        continue  # skip malformed
        return orgs

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching organizations for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
