from fastapi import FastAPI
from contextlib import asynccontextmanager
import polars as pl
import gcsfs
import os

if 'GOOGLE_APPLICATION_CREDENTIALS' not in os.environ:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\Romain\OneDrive - KU Leuven\focal-pager-460414-e9-45369b738be0.json"

bucket = "mda_eu_project"
path   = "data/consolidated_clean.parquet"
uri    = f"gs://{bucket}/{path}"

# Create a gcsfs filesystem (will use GOOGLE_APPLICATION_CREDENTIALS)
fs = gcsfs.GCSFileSystem()

# Eager load into memory:
df = pl.read_parquet(
    uri,
    storage_options={"gcs": fs}
)
print(df.head())