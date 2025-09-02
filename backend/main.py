from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn

# Global variables to store DataFrames
biorun_df = None
phylum_df = None

# threshold value
THRESHOLD = 100_000_000


def load_data():
    global biorun_df, phylum_df
    # load
    phylum_file_path = "data/sandpiper1.0.0.condensed.summary.phylum.csv.gz"
    biorun_file_path = "data/sandpiper1.0.0.condensed.biorun-metadata.csv.gz"

    columns_to_keep = ['run_accession', 'biosample', 'organism_name', 'run_total_bases']

    biorun_df = pd.read_csv(biorun_file_path, compression='gzip', index_col='run_accession', usecols=columns_to_keep)
    phylum_df = pd.read_csv(phylum_file_path, compression='gzip', index_col='biorun')

    # Remove rows where 'run_total_bases' < THRESHOLD
    biorun_df = biorun_df[biorun_df['run_total_bases'] >= THRESHOLD]

    # Remove duplications
    biorun_df = biorun_df.drop_duplicates()
    phylum_df = phylum_df.drop_duplicates()

    print("DataFrames loaded!")


def get_average_samples(organism_name: str):
    # Find runs for this organism
    accessions = biorun_df.loc[biorun_df["organism_name"] == organism_name].index

    if len(accessions) == 0:
        return {}  # no matches found

    # Filter phylum_df to only include those runs
    filtered_phylum = phylum_df.loc[phylum_df.index.intersection(accessions)]

    if filtered_phylum.empty:
        return {}

    # Replace 0 with NaN so they are ignored in the mean
    filtered_phylum = filtered_phylum.replace(0, pd.NA)

    # Compute average across phylum columns
    avg_phylum = filtered_phylum.mean()

    # Convert to dict and drop NaN/None values
    avg_phylum_dict = avg_phylum.dropna().to_dict()

    return avg_phylum_dict


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
def search_items(level: str, organism_name: str):
    result = get_average_samples(organism_name)
    return {"organism": organism_name, "level": level, "average": result}


if __name__ == "__main__":
    print("Starting FastAPI server on http://127.0.0.1:5002")
    uvicorn.run("main:app", host="127.0.0.1", port=5002, reload=True)