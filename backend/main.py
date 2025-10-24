from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, List
from pathlib import Path
import pandas as pd
import numpy as np
import uvicorn
import duckdb
import math

# ==========
# Response class: prefer ORJSON if installed, else fallback to JSON
# ==========
try:
    import orjson  # noqa: F401
    from fastapi.responses import ORJSONResponse as DefaultResponse
except Exception:
    from fastapi.responses import JSONResponse as DefaultResponse

# ========================
# Paths & config
# ========================

DATA_DIR = Path("data")
DB_PATH = Path("prepared/cache.duckdb")  # small on-disk DB; created on demand
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Input files (CSV.GZ)
PHYLUM_CSV = DATA_DIR / "sandpiper1.0.0.condensed.summary.phylum.csv.gz"
CLASS_CSV  = DATA_DIR / "sandpiper1.0.0.condensed.summary.class.csv.gz"
ORDER_CSV  = DATA_DIR / "sandpiper1.0.0.condensed.summary.order.csv.gz"
FAMILY_CSV = DATA_DIR / "sandpiper1.0.0.condensed.summary.family.csv.gz"
BIORUN_CSV = DATA_DIR / "sandpiper1.0.0.condensed.biorun-metadata.csv.gz"
BIOSMP_CSV = DATA_DIR / "sandpiper1.0.0.condensed.biosample-metadata.csv.gz"

# Threshold & geo
THRESHOLD = 100_000_000
RADIUS_KM = 100  # geospatial filter in kilometers

# Level -> config
LEVEL_CFG = {
    "phylum": {"csv": PHYLUM_CSV, "table": "taxon_phylum"},
    "class":  {"csv": CLASS_CSV,  "table": "taxon_class"},
    "order":  {"csv": ORDER_CSV,  "table": "taxon_order"},
    "family": {"csv": FAMILY_CSV, "table": "taxon_family"}
}

# Globals (small, in RAM)
biorun_df: Optional[pd.DataFrame] = None
biosample_df: Optional[pd.DataFrame] = None
duck: Optional[duckdb.DuckDBPyConnection] = None


# ========================
# Utilities
# ========================

def vectorized_parse_lat_lon(df: pd.DataFrame, col: str = "lat_lon") -> pd.DataFrame:
    """
    Robust, vectorized parser for common lat/lon formats:
      - "12.34 N 56.78 W"
      - "12.34, -56.78" or "12.34 -56.78"
    Writes float32 'latitude' and 'longitude'. Leaves NaN where unparsable.
    """
    s = df[col].astype("string")

    # A) With hemisphere letters
    pat_h = r'^\s*([+-]?\d+(?:\.\d+)?)\s*([NSns])\s+([+-]?\d+(?:\.\d+)?)\s*([EWew])\s*$'
    ex = s.str.extract(pat_h, expand=True)
    lat_h = pd.to_numeric(ex[0], errors="coerce")
    lon_h = pd.to_numeric(ex[2], errors="coerce")
    lat_h = lat_h.where(ex[1].str.upper().eq("N"), -lat_h)
    lon_h = lon_h.where(ex[3].str.upper().eq("E"), -lon_h)

    # B) Plain numbers (comma or space separated)
    pat_p = r'^\s*([+-]?\d+(?:\.\d+)?)\s*[,\s]\s*([+-]?\d+(?:\.\d+)?)\s*$'
    ex2 = s.str.extract(pat_p, expand=True)
    lat_p = pd.to_numeric(ex2[0], errors="coerce")
    lon_p = pd.to_numeric(ex2[1], errors="coerce")

    # Prefer hemisphere parse; fallback to plain numbers
    lat = lat_h.fillna(lat_p)
    lon = lon_h.fillna(lon_p)

    # If lat looks like longitude and vice versa, swap
    swap = (lat.abs() > 90) & (lon.abs() <= 90)
    lat = lon.where(swap, lat)
    lon = lat.where(swap, lon) if swap.any() else lon  # keeps original lon when no swap

    # Validate ranges
    lat = lat.where(lat.between(-90, 90))
    lon = lon.where(lon.between(-180, 180))

    df["latitude"] = lat.astype("float32")
    df["longitude"] = lon.astype("float32")

    # Optional: quick coverage log
    try:
        parsed = df["latitude"].notna() & df["longitude"].notna()
        print(f"[coords] parsed {parsed.sum():,} / {len(df):,} biosamples with coordinates")
    except Exception:
        pass

    return df


def haversine_distance(lat1, lon1, lat2_series, lon2_series):
    """Great-circle distance (km) between one point and arrays of points."""
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2_series.to_numpy())
    lon2 = np.radians(lon2_series.to_numpy())

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0
    return c * r


def bounding_box_prefilter(df: pd.DataFrame, lat: float, lon: float, radius_km: float) -> pd.DataFrame:
    """Cheap bounding-box filter to reduce Haversine calls."""
    if df.empty:
        return df
    deg_lat = radius_km / 111.0
    cos_lat = math.cos(math.radians(lat))
    deg_lon = radius_km / (111.0 * max(cos_lat, 1e-6))
    mask = (
        df["latitude"].between(lat - deg_lat, lat + deg_lat) &
        df["longitude"].between(lon - deg_lon, lon + deg_lon)
    )
    return df.loc[mask]


def os_cpu_count() -> int:
    try:
        import os
        return os.cpu_count() or 1
    except Exception:
        return 1


def connect_duck() -> duckdb.DuckDBPyConnection:
    """Open/create the on-disk DuckDB and set threads."""
    global duck
    if duck is None:
        duck = duckdb.connect(str(DB_PATH))
        duck.execute("PRAGMA threads=%d" % max(1, os_cpu_count()))
    return duck


def ensure_taxon_table(level: str) -> str:
    """
    Ensure a DuckDB table for the taxonomy level exists and is indexed on biorun.
    This runs once per level (first time you query that level).
    """
    cfg = LEVEL_CFG[level]
    table = cfg["table"]
    src = cfg["csv"]
    con = connect_duck()

    exists = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()
    if exists:
        return table

    print(f"[duckdb] creating table {table} from {src} ...")
    # read_csv_auto streams and infers types; sample_size increased for better inference
    con.execute(f"""
        CREATE TABLE "{table}" AS
        SELECT * FROM read_csv_auto(?, header=TRUE, sample_size=200000);
    """, [str(src)])
    print(f"[duckdb] created {table}")

    print(f"[duckdb] creating index on {table}(biorun) ...")
    con.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_biorun ON "{table}"(biorun);')
    print(f"[duckdb] index ready for {table}")

    return table


def duck_numeric_columns(con: duckdb.DuckDBPyConnection, table: str) -> List[str]:
    """Get numeric columns (except 'biorun') for a DuckDB table."""
    info = con.execute(f"PRAGMA table_info('{table}')").fetchdf()
    numeric_prefixes = (
        "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
        "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
        "REAL", "FLOAT", "DOUBLE", "DECIMAL"
    )
    cols = []
    for _, row in info.iterrows():
        name = str(row["name"])
        dtype = str(row["type"]).upper()
        if name.lower() == "biorun":
            continue
        if any(dtype.startswith(p) for p in numeric_prefixes):
            cols.append(name)
    return cols


def duck_avg_for_accessions(level: str, accessions: List[str]) -> Dict[str, float]:
    """
    Compute column-wise averages over selected taxon rows in DuckDB:
    - numeric columns only
    - treat 0 as missing (NULL)
    """
    if not accessions:
        return {}

    con = connect_duck()
    table = ensure_taxon_table(level)
    num_cols = duck_numeric_columns(con, table)
    if not num_cols:
        return {}

    select_expr = ", ".join([f'AVG(NULLIF("{c}", 0)) AS "{c}"' for c in num_cols])

    sql = f"""
        WITH acc AS (SELECT UNNEST(?) AS biorun)
        SELECT {select_expr}
        FROM "{table}" t
        JOIN acc USING (biorun)
    """
    df = con.execute(sql, [accessions]).fetchdf()
    if df.empty:
        return {}

    row = df.iloc[0]
    return {col: float(row[col]) for col in df.columns if pd.notna(row[col])}


# ========================
# Data loading (small only)
# ========================

def load_biorun_and_biosample() -> None:
    """Load only the smaller metadata CSVs into RAM (streaming parsers)."""
    global biorun_df, biosample_df

    biorun_dtypes = {
        "run_accession": "string",
        "biosample": "string",
        "organism_name": "string",
        "run_total_bases": "Int64",
    }

    print("[load] reading biorun ...")
    # Use pandas default engine (streaming) — avoids pyarrow MemoryError
    biorun_local = pd.read_csv(
        BIORUN_CSV,
        compression="gzip",
        usecols=list(biorun_dtypes.keys()),
        dtype=biorun_dtypes,
    )
    # Early filter to shrink rows
    biorun_local = biorun_local[biorun_local["run_total_bases"] >= THRESHOLD]
    # Category compress
    biorun_local["organism_name"] = biorun_local["organism_name"].astype("category")

    print("[load] reading biosample ...")
    biosample_local = pd.read_csv(
        BIOSMP_CSV,
        compression="gzip",
        usecols=["biosample", "lat_lon"],
        dtype={"biosample": "string", "lat_lon": "string"},
    )
    biosample_local = vectorized_parse_lat_lon(biosample_local, "lat_lon")
    biosample_local.drop(columns=["lat_lon"], inplace=True)

    print("[load] merging coords into biorun ...")
    biorun_local = biorun_local.merge(
        biosample_local,
        on="biosample",
        how="left",
        validate="m:1",
    )

    biorun_local.set_index("run_accession", inplace=True)
    biorun_local = biorun_local[~biorun_local.index.duplicated(keep="first")]

    # Make coords float32 to save memory
    for c in ("latitude", "longitude"):
        if c in biorun_local.columns:
            biorun_local[c] = biorun_local[c].astype("float32")

    # Expose the globals
    globals()["biorun_df"] = biorun_local
    globals()["biosample_df"] = biosample_local

    print(f"[load] biorun rows: {len(biorun_local):,}, organisms: {biorun_local['organism_name'].nunique():,}")


# ========================
# Core compute
# ========================

def compute_average(level: str, organism_name: str, lat: Optional[float], lon: Optional[float]) -> Dict[str, float]:
    """
    Narrow runs in-memory (organism + optional geofilter), then
    compute averages via indexed lookup in DuckDB.
    """
    # 1) Filter organism (category compare is fast)
    org_runs = biorun_df[biorun_df["organism_name"] == organism_name]
    if org_runs.empty:
        return {}

    # 2) Geo filter
    if lat is not None and lon is not None:
        org_runs = org_runs.dropna(subset=["latitude", "longitude"])
        if org_runs.empty:
            return {}
        org_runs = bounding_box_prefilter(org_runs, lat, lon, RADIUS_KM)
        if org_runs.empty:
            return {}
        d = haversine_distance(lat, lon, org_runs["latitude"], org_runs["longitude"])
        org_runs = org_runs.iloc[(d <= RADIUS_KM).nonzero()[0]]
        if org_runs.empty:
            return {}

    accessions = org_runs.index.astype(str).tolist()
    return duck_avg_for_accessions(level, accessions)


# ========================
# FastAPI app
# ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only small loads at startup
    load_biorun_and_biosample()
    # Open DuckDB (won't create big tables until first use)
    connect_duck()
    yield


app = FastAPI(lifespan=lifespan, default_response_class=DefaultResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in prod if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/samples/")
def search_items(level: str, organism_name: str, lat: Optional[float] = None, lon: Optional[float] = None):
    level_norm = level.lower()
    if level_norm not in LEVEL_CFG:
        return {"organism": organism_name, "level": level, "average": {}}

    try:
        result = compute_average(
            level_norm,
            str(organism_name),
            None if lat is None else float(lat),
            None if lon is None else float(lon),
        )
        return {"organism": organism_name, "level": level_norm, "average": result}
    except Exception as e:
        # Surface the error while you iterate
        return {"organism": organism_name, "level": level_norm, "average": {}, "error": str(e)}


if __name__ == "__main__":
    print("Starting FastAPI server on http://127.0.0.1:5002")
    # keep reload=False; reloader doubles memory use with big CSVs
    uvicorn.run("main:app", host="127.0.0.1", port=5002, reload=False)
