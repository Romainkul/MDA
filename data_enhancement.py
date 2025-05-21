import csv
import re
import polars as pl
from __future__ import annotations
import re, csv, pathlib, polars as pl

ROOT      = pathlib.Path(r"C:\Users\Romain\OneDrive - KU Leuven\Masters\MBIS\Year 2\Semester 2\Modern Data Analytics\CORDIS")
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
OUTDIR    = ROOT / "combined"
OUTDIR.mkdir(exist_ok=True)

###############################################################################
# 2.  Generic cleaner –– parameterised version of the loop you wrote
###############################################################################
_PROJECT_ID_RE = re.compile(r"^(?:19|20)\d{2}")
_GENERIC_NUM_RE  = re.compile(r"\d{4}")

import csv, pathlib, polars as pl, re

import csv, re, pathlib
import polars as pl                       #  >=0.20

import csv, pathlib, re
import polars as pl                       # ≥ 0.20


def _clean_one_file(csv_path: pathlib.Path,
                    number_regex: re.Pattern[str], dataset: str) -> pl.DataFrame:
    """
    Clean a CORDIS CSV whose long *objective* field sometimes explodes into
    extra columns because of stray quotes / semicolons.

    Strategy
    --------
    * A well-formed row has 21 semicolon-separated columns.
    * If we get more than 21 columns we treat columns 16 … -4 as belonging
      to *objective* and stitch them back together with a semicolon.
    * The last three columns are   contentUpdateDate | rcn | grantDoi.
    """
    # ---------- constants --------------------------------------------------
    if dataset=="project":
        EXPECTED_COLS   = 20          # final width
        TITLE_COL       = 3           # 0-based index of *title*
        DATE1_COL       = 4           # 0-based index of startDate
        DATE2_COL       = 5           # 0-based index of endDate
        OBJECTIVE_COL   = 16          # 0-based index of objective
        TRAILING_KEEP   = 3           # last three fixed columns
    elif dataset=="organization":
        EXPECTED_COLS   = 25          # final width
        TITLE_COL       = 3           # 0-based index of *title*
        DATE1_COL       = 4           # 0-based index of startDate
        DATE2_COL       = 5           # 0-based index of endDate
        OBJECTIVE_COL   = 4           # 0-based index of objective
        TRAILING_KEEP   = 20           # last three fixed columns
    else:
        EXPECTED_COLS   = 20          # final width
        TITLE_COL       = 3           # 0-based index of *title*
        DATE1_COL       = 4           # 0-based index of startDate
        DATE2_COL       = 5           # 0-based index of endDate
        OBJECTIVE_COL   = 16          # 0-based index of objective
        TRAILING_KEEP   = 3           # last three fixed columns



    date_rx   = re.compile(r"\d{4}-\d{2}-\d{2}$")
    is_date   = lambda s: (s == "") or bool(date_rx.match(s))

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

        # ---------- iterate raw lines -------------------------------------
        for raw in fin:
            #print(raw)
            raw = raw.rstrip("\n")
            #print(raw)
            cells = raw.split(";")                     # blind split

            # ---- 1️⃣  repair *title* if dates are not where they belong --
            if (len(cells) > EXPECTED_COLS) and  (not is_date(cells[DATE1_COL]) or not is_date(cells[DATE2_COL])) and dataset=="project":
                # look for the first position where *two successive* cells
                # are both valid dates / nulls
                i = DATE1_COL
                while i + 1 < len(cells):
                    if is_date(cells[i]) and is_date(cells[i + 1]):
                        break
                    i += 1
                else:
                    # cannot find a valid date pair → give up on this line
                    continue

                head   = cells[:TITLE_COL]             # 0 … 2
                title  = ";".join(cells[TITLE_COL:i])  # glue spill-over
                cells  = head + [title] + cells[i:]    # rebuild the row
            # ---- 2️⃣  repair *objective* overflow ------------------------
            if len(cells) > EXPECTED_COLS and (dataset=="project" or  dataset=="organization"):
                head = cells[:OBJECTIVE_COL]
                tail = cells[-TRAILING_KEEP:]
                obj  = ";".join(cells[OBJECTIVE_COL:-TRAILING_KEEP])
                cells = head + [obj] + tail
                #print("here 2")

            # ---- 3️⃣  pad short rows, skip malformed ---------------------
            if len(cells) < EXPECTED_COLS and (dataset=="project" or  dataset=="organization"):
                cells.extend([""] * (EXPECTED_COLS - len(cells)))
                #print("here again")

            if len(cells) != EXPECTED_COLS and (dataset=="project" or  dataset=="organization"):            # still wrong → skip
                #print(cells)
                continue

            # ---- 4️⃣  cell-level clean-ups -------------------------------
            cleaned: list[str] = []
            for cell in cells:

                if cell in ('""', ""):
                    cell = ""
                else:
                    cell = (cell.replace("\t", " ")
                                 .replace('"""', '"')
                                 .strip())
                    if number_regex.fullmatch(cell):
                        cell = cell.lstrip("0") or "0"
                cleaned.append(cell.strip('"'))
            cleaned[-1]=cleaned[-1].replace('"','').replace(',','')
            cleaned[0]=cleaned[0].replace('"','')
            writer.writerow(cleaned)

    # ---------- read into Polars (all Utf8) -------------------------------
    return pl.read_csv(
        tmp_clean,
        separator="|",
        quote_char='"',
        has_header=True,
        infer_schema_length=0,
        null_values=[""],
        truncate_ragged_lines=True,
    )


def combine_all_programmes() -> None:
    from pathlib import Path
    for dataset in DATASETS:
        combined: list[pl.DataFrame] = []

        for i,programme_dir in enumerate(ROOT.iterdir()):
            if not programme_dir.is_dir():
                continue
            csv_file = programme_dir / f"{dataset}.csv"
            if not csv_file.exists():
                continue

            regex = _PROJECT_ID_RE if dataset == "project" else _GENERIC_NUM_RE
            df    = _clean_one_file(csv_file, regex, dataset)
            print(programme_dir)
            # ---------- type coercions matching your original code ----------
            if dataset == "project":
                df = (
                    df
                    .with_columns([
                        pl.col("id"),#.cast(pl.Int64),
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
                        pl.col("totalCost").cast(pl.Utf8, strict=False).str.strip_chars('"').str.replace_all('"','').str.replace(",",".").cast(pl.Float64),
                        pl.col("ecMaxContribution").cast(pl.Utf8, strict=False).str.strip_chars('"').str.replace_all('"','').str.replace(",",".").cast(pl.Float64),
                        pl.col("startDate").cast(pl.Utf8, strict=False).str.strip_chars('"').str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                        pl.col("endDate").cast(pl.Utf8, strict=False).str.strip_chars('"').str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                        pl.col("ecSignatureDate").cast(pl.Utf8, strict=False).str.strip_chars('"').str.strptime(pl.Date, "%Y-%m-%d", strict=False),
                        pl.col("contentUpdateDate").cast(pl.Utf8, strict=False).str.strip_chars('"').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                        pl.col("rcn").cast(pl.Int64),
                    ])
                    .with_columns(
                        pl.lit(programme_dir.name).alias("programmeFolder")   # <-- NEW COLUMN
                    )
                )
            elif dataset == "organization":
                df = df.with_columns([
                    pl.col("contentUpdateDate").cast(pl.Utf8, strict=False).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                    pl.col("totalCost").cast(pl.Utf8, strict=False).str.replace(",",".").cast(pl.Float64),
                ])
            elif dataset == "projectDeliverables":
                df = df.with_columns([
                    #pl.col("projectID").cast(pl.Int64),
                    pl.col("contentUpdateDate").cast(pl.Utf8, strict=False)
                    .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                ])
            elif dataset == "projectPublications":
                if programme_dir==Path(r"C:\Users\Romain\OneDrive - KU Leuven\Masters\MBIS\Year 2\Semester 2\Modern Data Analytics\CORDIS\H2013"):
                    rename_map = {
                        "RECORD_ID":      "id",
                        "TITLE":          "title",
                        "AUTHOR":         "authors",
                        "DOI":            "doi",
                        "PROJECT_ID":     "projectID",
                        "JOURNAL_TITLE":  "journalTitle",
                        "PAGES":          "publishedPages",
                        "PUBLICATION_TYPE": "isPublishedAs",
                    }

                    df = df.rename(rename_map)
                else:
                    df = df.with_columns([
                        pl.col("contentUpdateDate").cast(pl.Utf8, strict=False)
                        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                        pl.col("id").cast(pl.Utf8, strict=False)
                        .str.extract(r"^(\d+)_", 1)
                        #.cast(pl.Int64)
                        .alias("projectID"),
                    ])
            elif dataset == "reportSummaries":
                df = df.with_columns(
                    pl.col("contentUpdateDate").cast(pl.Utf8, strict=False)
                    .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                )
            elif dataset == "organization":
                df = df.with_columns([
                    pl.col("contentUpdateDate").cast(pl.Utf8, strict=False)
                    .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False),
                    pl.col("totalCost").cast(pl.Utf8, strict=False)
                    .str.replace(",", ".")
                    .cast(pl.Float64),
                ])
            elif dataset == "webItem":
                df = df.with_columns(
                    pl.col("uri").cast(pl.Utf8, strict=False)
                    .str.extract(r"/files/\d+/(\d+)/", 1)
                    .cast(pl.Int64)
                    .alias("projectID"),
                )

            # ---------------------------------------------------------------
            combined.append(df)

        # --------------------------------------------------------------------
        # Write out per-dataset parquet
        # --------------------------------------------------------------------
        if combined:
            how="vertical_relaxed"
            if dataset=="projectPublications":
                how="diagonal"
            result = pl.concat(combined, how=how)
            parquet_path = OUTDIR / f"{dataset}_all.parquet"
            result.write_parquet(parquet_path)
            print(f"✔  {dataset:15s} → {parquet_path}")

import pathlib
import polars as pl

ROOT    = pathlib.Path(r"C:\Users\Romain\OneDrive - KU Leuven\Masters\MBIS\Year 2\Semester 2\Modern Data Analytics\CORDIS")
OUTDIR  = ROOT / "combined"
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

dfs = {}
for dataset in DATASETS:
    path = OUTDIR / f"{dataset}_all.parquet"
    dfs[dataset] = pl.read_parquet(path)

projects         = dfs["project"]

projects_deliv   = (
    dfs["projectDeliverables"]
    .group_by("projectID")
    .agg([
        pl.col("deliverableType").alias("list_deliverableType"),
        pl.col("url")            .alias("list_url"),
        pl.col("contentUpdateDate").alias("list_contentUpdateDate"),
    ])
)

projects_publi   = (
    dfs["projectPublications"]
    .group_by("projectID")
    .agg([
        pl.col("authors")         .alias("list_authors"),
        pl.col("title")           .alias("list_title"),
        pl.col("doi")             .alias("list_doi"),
        pl.col("journalTitle")    .alias("list_journalTitle"),
        pl.col("isPublishedAs")   .alias("list_isPublishedAs"),
        pl.col("publishedYear")   .alias("list_publishedYear"),
        pl.col("contentUpdateDate").alias("list_contentUpdateDate"),
    ])
)

report = (
    dfs["reportSummaries"]
    .group_by("projectID")
    .agg([
        pl.col("title")           .alias("list_title"),
        pl.col("attachment")      .alias("list_attachment"),
        pl.col("contentUpdateDate").alias("list_contentUpdateDate"),
    ])
)

org = (
    dfs["organization"]
    .group_by("projectID")
    .agg([
        pl.col("organisationID")  .alias("list_organisationID"),
        pl.col("country")         .alias("list_country"),
        pl.col("name")            .alias("list_name"),
        pl.col("SME")             .alias("list_SME"),
        pl.col("city")            .alias("list_city"),
        pl.col("geolocation")     .alias("list_geolocation"),
        pl.col("organizationURL") .alias("list_organizationURL"),
        pl.col("role")            .alias("list_role"),
        pl.col("ecContribution")  .alias("list_ecContribution"),
        pl.col("netEcContribution").alias("list_netEcContribution"),
        pl.col("totalCost")       .alias("list_totalCost"),
        pl.col("endOfParticipation").alias("list_endOfParticipation"),
        pl.col("activityType")    .alias("list_activityType"),
        pl.col("contentUpdateDate").alias("list_contentUpdateDate"),
    ])
)

voc = (
    dfs["euroSciVoc"]
    .group_by("projectID")
    .agg([
        pl.col("euroSciVocTitle")      .alias("list_euroSciVocTitle"),
        pl.col("euroSciVocPath")       .alias("list_euroSciVocPath"),
        pl.col("euroSciVocDescription").alias("list_description"),
    ])
)

topic = (
    dfs["topics"]
    .group_by("projectID")
    .agg([
        pl.col("topic")   .alias("list_topic"),
        pl.col("title")   .alias("list_title"),
    ])
)

web_item = dfs["webItem"]  # no aggregation

web_link = (
    dfs["webLink"]
    .group_by("projectID")
    .agg([
        pl.col("physUrl")            .alias("list_physUrl"),
        pl.col("availableLanguages") .alias("list_availableLanguages"),
        pl.col("status")             .alias("list_status"),
        pl.col("archivedDate")       .alias("list_archivedDate"),
        pl.col("type")               .alias("list_type"),
        pl.col("source")             .alias("list_source"),
        pl.col("represents")         .alias("list_represents"),
    ])
)

legal = (
    dfs["legalBasis"]
    .group_by("projectID")
    .agg([
        pl.col("legalBasis")         .alias("list_legalBasis"),
        pl.col("title")              .alias("list_title"),
        pl.col("uniqueProgrammePart").alias("list_uniqueProgrammePart"),
    ])
)

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

for col in ["startDate", "endDate"]:
    if consolidated[col].dtype == pl.Utf8:
        consolidated = consolidated.with_column(
            pl.col(col).str.strptime(pl.Date, "%Y-%m-%d").alias(col)
        )

consolidated = consolidated.with_columns(
    pl.col("list_netEcContribution").list.eval(pl.element().cast(pl.Float64),parallel=True)
    .list.sum().alias("netEcContribution")
)

consolidated = consolidated.with_columns(
    pl.col("totalCost").cast(pl.Float64),
    pl.col("netEcContribution").cast(pl.Float64)
)

consolidated = consolidated.with_columns([
    pl.col("startDate").dt.year().alias("startYear"),
    pl.col("endDate").  dt.year().alias("endYear"),
    (pl.col("endDate") - pl.col("startDate")).dt.total_days().alias("durationDays"),
    (pl.col("netEcContribution") / pl.col("totalCost")).alias("ecRatio"),
])

consolidated.write_parquet(OUTDIR / "consolidated.parquet")

excluded_frameworks = ["FP1", "FP2", "FP3", "FP4", "FP5", "FP6"]

consolidated_clean = (consolidated.filter(~pl.col("frameworkProgramme").is_in(excluded_frameworks)))

consolidated_clean.write_parquet(OUTDIR / "consolidated_clean.parquet")
