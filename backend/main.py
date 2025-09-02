from fastapi import FastAPI
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


load_data()

app = FastAPI()


@app.get("/samples/")
def search_items(level: str, organism_name: str):
    # TODO: implement it!
    pass


if __name__ == "__main__":
    print("Starting FastAPI server on http://127.0.0.1:5000")
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)