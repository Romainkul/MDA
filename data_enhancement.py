"""
CORDIS Data Cleaning and Consolidation Script

- Cleans "dirty" CORDIS CSVs with misaligned columns due to text fields.
- Type-coerces and merges per-dataset data across all available programme folders.
- Aggregates project-related data into a single consolidated Parquet file.
"""

import re
import csv
import pathlib
import polars as pl

# ==== PATHS AND DATASET CONFIGURATION ========================================

ROOT = pathlib.Path(
    r"C:\Users\Romain\OneDrive - KU Leuven\Masters\MBIS\Year 2\Semester 2\Modern Data Analytics\CORDIS"
)
OUTDIR = ROOT / "combined"
OUTDIR.mkdir(exist_ok=True)

DATASETS = [
    "project",
    "projectDeliverables",
    "projectPublications",
    "reportSummaries",
    "organization",
    "euroSciVoc",
    "topics",
    "webItem",
    "webLink",
    "legalBasis",
]

# ==== REGEX FOR CLEANING =====================================================
_PROJECT_ID_RE = re.compile(r"^(?:19|20)\d{2}")
_GENERIC_NUM_RE = re.compile(r"\d{4}")

# ==== CLEANING FUNCTION ======================================================

def _clean_one_file(csv_path: pathlib.Path, number_regex: re.Pattern, dataset: str) -> pl.DataFrame:
    """
    Cleans a CORDIS CSV file, handling column misalignment due to unescaped semicolons.

    Args:
        csv_path: Path to input CSV file.
        number_regex: Regex to clean numeric fields.
        dataset: Name of the dataset ("project", "organization", etc.)

    Returns:
        Cleaned Polars DataFrame.
    """

    # Dataset-specific expected column settings
    dataset_settings = {
        "project":        dict(EXPECTED_COLS=20, OBJECTIVE_COL=16, TRAILING_KEEP=3),
        "organization":   dict(EXPECTED_COLS=25, OBJECTIVE_COL=4,  TRAILING_KEEP=20),
    }
    DEFAULT_SETTINGS = dict(EXPECTED_COLS=20, OBJECTIVE_COL=16, TRAILING_KEEP=3)
    settings = dataset_settings.get(dataset, DEFAULT_SETTINGS)

    EXPECTED_COLS = settings["EXPECTED_COLS"]
    OBJECTIVE_COL = settings["OBJECTIVE_COL"]
    TRAILING_KEEP = settings["TRAILING_KEEP"]

    date_rx = re.compile(r"\d{4}-\d{2}-\d{2}$")
    is_date = lambda s: (s == "") or bool(date_rx.match(s))
    tmp_clean = csv_path.with_suffix(".cleaned.csv")

    with csv_path.open(encoding="utf-8", newline="") as fin, \
         tmp_clean.open("w", encoding="utf-8", newline="") as fout:

        writer = csv.writer(
            fout,
            delimiter="|",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )

        for raw in fin:
            raw = raw.rstrip("\n")
            cells = raw.split(";")  # Naive split

            # Step 1: Repair "title" overflow due to misaligned dates (for "project" only)
            if (
                dataset == "project" and len(cells) > EXPECTED_COLS
                and (not is_date(cells[4]) or not is_date(cells[5]))
            ):
                i = 4
                while i + 1 < len(cells):
                    if is_date(cells[i]) and is_date(cells[i + 1]):
                        break
                    i += 1
                else:
                    continue  # Skip if not fixable

                head = cells[:3]
                title = ";".join(cells[3:i])
                cells = head + [title] + cells[i:]

            # Step 2: Repair "objective" overflow (for project/organization)
            if len(cells) > EXPECTED_COLS and dataset in ("project", "organization"):
                head = cells[:OBJECTIVE_COL]
                tail = cells[-TRAILING_KEEP:]
                obj = ";".join(cells[OBJECTIVE_COL:-TRAILING_KEEP])
                cells = head + [obj] + tail

            # Step 3: Pad short rows, skip if still malformed
            if len(cells) < EXPECTED_COLS and dataset in ("project", "organization"):
                cells.extend([""] * (EXPECTED_COLS - len(cells)))
            if len(cells) != EXPECTED_COLS and dataset in ("project", "organization"):
                continue

            # Step 4: Cell-level cleaning
            cleaned = []
            for cell in cells:
                cell = cell.strip('"')
                if cell in ('""', ""):
                    cell = ""
                else:
                    cell = cell.replace("\t", " ").replace('"""', '"').strip()
                    if number_regex.fullmatch(cell):
                        cell = cell.lstrip("0") or "0"
                cleaned.append(cell)
            if cleaned:
                cleaned[-1] = cleaned[-1].replace('"', '').replace(',', '')
                cleaned[0] = cleaned[0].replace('"', '')
            writer.writerow(cleaned)

    # Read cleaned file with Polars
    return pl.read_csv(
        tmp_clean,
        separator="|",
        quote_char='"',
        has_header=True,
        infer_schema_length=0,
        null_values=[""],
        truncate_ragged_lines=True,
    )

# ==== COMBINING AND TYPE CASTING ACROSS PROGRAMMES ===========================

def combine_all_programmes() -> None:
    """
    Combines and cleans each CORDIS dataset across all available programmes,
    and writes a single Parquet file per dataset.
    """
    for dataset in DATASETS:
        combined = []
        for programme_dir in ROOT.iterdir():
            if not programme_dir.is_dir():
                continue
            csv_file = programme_dir / f"{dataset}.csv"
            if not csv_file.exists():
                continue

            regex = _PROJECT_ID_RE if dataset == "project" else _GENERIC_NUM_RE
            df = _clean_one_file(csv_file, regex, dataset)

            # Type coercions (dataset-specific)
            if dataset == "project":
                df = (
                    df.with_columns([
                        pl.col("id"),
                        pl.col("acronym").cast(pl.Utf8, strict=False).str.strip_chars('"'),
                        pl.col("status").cast(pl.Utf8, strict=False).str.strip_chars('"'),
                        pl.col("title").cast(pl.Utf8, strict=False).str.strip_chars('"'),
                        pl.col("legalBasis").cast(pl.Utf8, strict=False).str.strip_chars('"'),
                        pl.col("topics").cast(pl.Utf8, strict=False).str.strip_chars('"'),
                        pl.col("frameworkProgramme").cast(pl.Utf8, strict=False).str.strip_chars('"'),
                        pl.col("masterCall").cast(pl.Utf8, strict=False).str.strip_chars('"'),
                        pl.col("subCall").cast(pl.Utf8, strict=False).str.strip_chars('"'),
                        pl.col("fundingScheme").cast(pl.Utf8, strict=False).str.strip_chars('"'),
                        pl.col("nature").cast(pl.Utf8, strict=False).str.strip_chars('"'),
                        pl.col("objective").cast(pl.Utf8, strict=False).str.strip_chars('"'),
                        pl.col("grantDoi").cast(pl.Utf8, strict=False).str.strip_chars('"'),
                        pl.col("totalCost").cast(pl.Utf8, strict=False).str.strip_chars('"').str.replace_all('"', '').str.replace(",", ".").cast(pl.Float64),
                        pl.col("ecMaxContribution").cast(pl.Utf8, strict=False).str.strip_chars('"').str.replace_all('"', '').str.replace(",", ".").cast(pl.Float64),
                        pl.col("startDate").cast(pl.Utf8, strict=False).str.strip_chars('"').str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                        pl.col("endDate").cast(pl.Utf8, strict=False).str.strip_chars('"').str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                        pl.col("ecSignatureDate").cast(pl.Utf8, strict=False).str.strip_chars('"').str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                        pl.col("contentUpdateDate").cast(pl.Utf8, strict=False).str.strip_chars('"').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                        pl.col("rcn").cast(pl.Int64),
                    ])
                    .with_columns(
                        pl.lit(programme_dir.name).alias("programmeFolder")
                    )
                )
            elif dataset == "organization":
                df = df.with_columns([
                    pl.col("contentUpdateDate").cast(pl.Utf8, strict=False).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                    pl.col("totalCost").cast(pl.Utf8, strict=False).str.replace(",", ".").cast(pl.Float64),
                ])
            elif dataset == "projectDeliverables":
                df = df.with_columns([
                    pl.col("contentUpdateDate").cast(pl.Utf8, strict=False).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                ])
            elif dataset == "projectPublications":
                # Special handling for H2013, else standardize
                if programme_dir.name == "H2013":
                    rename_map = {
                        "RECORD_ID": "id",
                        "TITLE": "title",
                        "AUTHOR": "authors",
                        "DOI": "doi",
                        "PROJECT_ID": "projectID",
                        "JOURNAL_TITLE": "journalTitle",
                        "PAGES": "publishedPages",
                        "PUBLICATION_TYPE": "isPublishedAs",
                    }
                    df = df.rename(rename_map)
                else:
                    df = df.with_columns([
                        pl.col("contentUpdateDate").cast(pl.Utf8, strict=False).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                        pl.col("id").cast(pl.Utf8, strict=False).str.extract(r"^(\d+)_", 1).alias("projectID"),
                    ])
            elif dataset == "reportSummaries":
                df = df.with_columns(
                    pl.col("contentUpdateDate").cast(pl.Utf8, strict=False).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                )
            elif dataset == "webLink":
                pass  # No special handling
            elif dataset == "webItem":
                df = df.with_columns(
                    pl.col("uri").cast(pl.Utf8, strict=False).str.extract(r"/files/\d+/(\d+)/", 1).cast(pl.Int64).alias("projectID"),
                )
            elif dataset == "legalBasis":
                pass  # No special handling

            combined.append(df)

        # Write out per-dataset Parquet
        if combined:
            how = "diagonal" if dataset == "projectPublications" else "vertical_relaxed"
            result = pl.concat(combined, how=how)
            parquet_path = OUTDIR / f"{dataset}_all.parquet"
            result.write_parquet(parquet_path)
            print(f"✔  {dataset:20s} → {parquet_path}")

# ==== CALL THE COMBINER FUNCTION TO GENERATE PARQUETS ========================

combine_all_programmes()

# ==== AGGREGATION AND CONSOLIDATION ==========================================

# Load all combined Parquet files
dfs = {dataset: pl.read_parquet(OUTDIR / f"{dataset}_all.parquet") for dataset in DATASETS}

# Aggregate per-project lists
projects = dfs["project"]

projects_deliv = dfs["projectDeliverables"].group_by("projectID").agg([
    pl.col("deliverableType").alias("list_deliverableType"),
    pl.col("url").alias("list_url"),
    pl.col("contentUpdateDate").alias("list_contentUpdateDate"),
])
projects_publi = dfs["projectPublications"].group_by("projectID").agg([
    pl.col("authors").alias("list_authors"),
    pl.col("title").alias("list_title"),
    pl.col("doi").alias("list_doi"),
    pl.col("journalTitle").alias("list_journalTitle"),
    pl.col("isPublishedAs").alias("list_isPublishedAs"),
    pl.col("publishedYear").alias("list_publishedYear"),
    pl.col("contentUpdateDate").alias("list_contentUpdateDate"),
])
report = dfs["reportSummaries"].group_by("projectID").agg([
    pl.col("title").alias("list_title"),
    pl.col("attachment").alias("list_attachment"),
    pl.col("contentUpdateDate").alias("list_contentUpdateDate"),
])
org = dfs["organization"].group_by("projectID").agg([
    pl.col("organisationID").alias("list_organisationID"),
    pl.col("country").alias("list_country"),
    pl.col("name").alias("list_name"),
    pl.col("SME").alias("list_SME"),
    pl.col("city").alias("list_city"),
    pl.col("geolocation").alias("list_geolocation"),
    pl.col("organizationURL").alias("list_organizationURL"),
    pl.col("role").alias("list_role"),
    pl.col("ecContribution").alias("list_ecContribution"),
    pl.col("netEcContribution").alias("list_netEcContribution"),
    pl.col("totalCost").alias("list_totalCost"),
    pl.col("endOfParticipation").alias("list_endOfParticipation"),
    pl.col("activityType").alias("list_activityType"),
    pl.col("contentUpdateDate").alias("list_contentUpdateDate"),
])
voc = dfs["euroSciVoc"].group_by("projectID").agg([
    pl.col("euroSciVocTitle").alias("list_euroSciVocTitle"),
    pl.col("euroSciVocPath").alias("list_euroSciVocPath"),
    pl.col("euroSciVocDescription").alias("list_description"),
])
topic = dfs["topics"].group_by("projectID").agg([
    pl.col("topic").alias("list_topic"),
    pl.col("title").alias("list_title"),
])
web_item = dfs["webItem"]
web_link = dfs["webLink"].group_by("projectID").agg([
    pl.col("physUrl").alias("list_physUrl"),
    pl.col("availableLanguages").alias("list_availableLanguages"),
    pl.col("status").alias("list_status"),
    pl.col("archivedDate").alias("list_archivedDate"),
    pl.col("type").alias("list_type"),
    pl.col("source").alias("list_source"),
    pl.col("represents").alias("list_represents"),
])
legal = dfs["legalBasis"].group_by("projectID").agg([
    pl.col("legalBasis").alias("list_legalBasis"),
    pl.col("title").alias("list_title"),
    pl.col("uniqueProgrammePart").alias("list_uniqueProgrammePart"),
])

# Join all aggregated info into a consolidated DataFrame
consolidated = (
    projects
    .join(projects_deliv,   left_on="id", right_on="projectID", suffix="_deliv", how="left")
    .join(projects_publi,   left_on="id", right_on="projectID", suffix="_publi", how="left")
    .join(report,           left_on="id", right_on="projectID", suffix="_report", how="left")
    .join(org,              left_on="id", right_on="projectID", suffix="_org", how="left")
    .join(web_link,         left_on="id", right_on="projectID", suffix="_link", how="left")
    .join(legal,            left_on="id", right_on="projectID", suffix="_legal", how="left")
    .join(topic,            left_on="id", right_on="projectID", suffix="_topic", how="left")
    .join(voc,              left_on="id", right_on="projectID", suffix="_voc", how="left")
)

# Standardize dates and compute extra fields
for col in ["startDate", "endDate"]:
    if consolidated[col].dtype == pl.Utf8:
        consolidated = consolidated.with_column(
            pl.col(col).str.strptime(pl.Date, "%Y-%m-%d").alias(col)
        )

consolidated = consolidated.with_columns([
    pl.col("list_netEcContribution").list.eval(pl.element().cast(pl.Float64), parallel=True).list.sum().alias("netEcContribution"),
    pl.col("totalCost").cast(pl.Float64),
    pl.col("startDate").dt.year().alias("startYear"),
    pl.col("endDate").dt.year().alias("endYear"),
    (pl.col("endDate") - pl.col("startDate")).dt.total_days().alias("durationDays"),
])

consolidated = consolidated.with_columns([
        (pl.col("netEcContribution") / pl.col("totalCost")).alias("ecRatio"),
])

consolidated.write_parquet(OUTDIR / "consolidated.parquet")
print(f"✔  consolidated → {OUTDIR / 'consolidated.parquet'}")

# ==== CLEANING FILTERS =======================================================

excluded_frameworks = ["FP1", "FP2", "FP3", "FP4", "FP5", "FP6"]
consolidated_clean = consolidated.filter(~pl.col("frameworkProgramme").is_in(excluded_frameworks))
consolidated_clean.write_parquet(OUTDIR / "consolidated_clean.parquet")
print(f"✔  consolidated_clean → {OUTDIR / 'consolidated_clean.parquet'}")
