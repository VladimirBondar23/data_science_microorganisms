from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import uvicorn
from typing import Optional

# Global variables to store DataFrames
biorun_df = None
biosample_df = None
phylum_df = None
order_df = None
family_df = None
class_df = None
genus_df = None


# threshold value
THRESHOLD = 100_000_000
RADIUS_KM = 100  # Radius for geospatial filtering in kilometers


def parse_lat_lon(lat_lon_str):
    """
    Parses a latitude/longitude string and returns a tuple of (latitude, longitude).
    Handles various formats and missing values.
    """
    if pd.isna(lat_lon_str) or not isinstance(lat_lon_str, str):
        return np.nan, np.nan

    lat_lon_str = lat_lon_str.strip().lower()
    if lat_lon_str in ['missing', 'not applicable']:
        return np.nan, np.nan

    parts = lat_lon_str.split()
    try:
        if len(parts) == 4:
            lat = float(parts[0])
            if parts[1].lower() == 's':
                lat = -lat

            lon = float(parts[2])
            if parts[3].lower() == 'w':
                lon = -lon

            return lat, lon
        else:
            return np.nan, np.nan
    except (ValueError, IndexError):
        return np.nan, np.nan


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r


def load_data():
    global biorun_df, phylum_df, biosample_df, order_df, family_df, class_df, genus_df
    # File paths
    phylum_file_path = "data/sandpiper1.0.0.condensed.summary.phylum.csv.gz"
    biorun_file_path = "data/sandpiper1.0.0.condensed.biorun-metadata.csv.gz"
    biosample_file_path = "data/sandpiper1.0.0.condensed.biosample-metadata.csv.gz"
    order_file_path = "data/sandpiper1.0.0.condensed.summary.order.csv.gz"
    family_file_path = "data/sandpiper1.0.0.condensed.summary.family.csv.gz"
    class_file_path = "data/sandpiper1.0.0.condensed.summary.class.csv.gz"
    genus_file_path = "data/sandpiper1.0.0.condensed.summary.genus.csv.gz"

    # Columns to keep from biorun metadata
    columns_to_keep_biorun = ['run_accession', 'biosample', 'organism_name', 'run_total_bases']

    biorun_df = pd.read_csv(biorun_file_path, compression='gzip', usecols=columns_to_keep_biorun)
    phylum_df = pd.read_csv(phylum_file_path, compression='gzip', index_col='biorun')
    class_df = pd.read_csv(class_file_path, compression='gzip', index_col='biorun')
    order_df = pd.read_csv(order_file_path, compression='gzip', index_col='biorun')


    # --- OPTIMIZATION ---
    # Only load 'biosample' and 'lat_lon' columns to save memory
    columns_to_keep_biosample = ['biosample', 'lat_lon']
    biosample_df = pd.read_csv(biosample_file_path, compression='gzip', usecols=columns_to_keep_biosample)

    # Parse the lat_lon column
    coords = biosample_df['lat_lon'].apply(parse_lat_lon)
    biosample_df[['latitude', 'longitude']] = pd.DataFrame(coords.tolist(), index=biosample_df.index)

    # Merge biorun_df with the lean biosample_df
    biorun_df = pd.merge(
        biorun_df,
        biosample_df[['biosample', 'latitude', 'longitude']],
        on='biosample',
        how='left'
    )
    biorun_df.set_index('run_accession', inplace=True)

    # Filter by base count
    biorun_df = biorun_df[biorun_df['run_total_bases'] >= THRESHOLD]

    # Remove duplicates
    biorun_df = biorun_df.drop_duplicates()
    phylum_df = phylum_df.drop_duplicates()
    class_df = class_df.drop_duplicates()
    order_df = order_df.drop_duplicates()

    print("DataFrames loaded efficiently and merged!")


def get_average_samples(level: str, organism_name: str, lat: Optional[float] = None, lon: Optional[float] = None):

    taxon_data_frame = ''
    if level.lower() == 'phylum':
        taxon_data_frame = phylum_df
    elif level.lower() == 'class':
        taxon_data_frame = class_df
    elif level.lower() == 'order':
        taxon_data_frame = order_df

    # Start with the full dataset for the organism
    organism_runs = biorun_df[biorun_df["organism_name"] == organism_name]

    # If lat and lon are provided, filter by location
    if lat is not None and lon is not None:
        # Drop rows with no coordinate data
        organism_runs = organism_runs.dropna(subset=['latitude', 'longitude'])

        if not organism_runs.empty:
            # Calculate distances
            distances = haversine_distance(lat, lon, organism_runs['latitude'], organism_runs['longitude'])

            # Filter by radius
            organism_runs = organism_runs[distances <= RADIUS_KM]

    accessions = organism_runs.index

    if len(accessions) == 0:
        return {}  # no matches found

    # Filter to only include those runs
    filtered_taxon = taxon_data_frame.loc[taxon_data_frame.index.intersection(accessions)]

    if filtered_taxon.empty:
        return {}

    # Drop the 'biorun' column before calculating the mean
    # Check if 'biorun' column exists before dropping
    if 'biorun' in filtered_taxon.columns:
        filtered_taxon = filtered_taxon.drop(columns=['biorun'])

    # Replace 0 with NaN so they are ignored in the mean
    filtered_taxon = filtered_taxon.replace(0, pd.NA)

    # Compute average across phylum columns
    avg_taxon = filtered_taxon.mean()

    # Convert to dict and drop NaN/None values
    avg_taxon_dict = avg_taxon.dropna().to_dict()

    return avg_taxon_dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_data()
    yield


app = FastAPI(lifespan=lifespan)

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/samples/")
def search_items(level: str, organism_name: str, lat: Optional[float] = None, lon: Optional[float] = None):
    result = get_average_samples(level, organism_name, lat, lon)
    return {"organism": organism_name, "level": level, "average": result}


if __name__ == "__main__":
    print("Starting FastAPI server on http://127.0.0.1:5002")
    uvicorn.run("main:app", host="127.0.0.1", port=5002, reload=True)