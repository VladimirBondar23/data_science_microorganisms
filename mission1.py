import pandas as pd

# load
phylum_file_path = "/content/sandpiper1.0.0.condensed.summary.phylum.csv.gz"
biorun_file_path = "/content/sandpiper1.0.0.condensed.biorun-metadata.csv.gz"

columns_to_keep = ['run_accession', 'biosample', 'organism_name', 'run_total_bases']

biorun_df = pd.read_csv(biorun_file_path, compression='gzip', index_col='run_accession', usecols=columns_to_keep)
phylum_df = pd.read_csv(phylum_file_path, compression='gzip', index_col='biorun')

# filter
THRESHOLD = 100_000_000  # threshold value

# Remove rows where 'run_total_bases' < THRESHOLD
biorun_df = biorun_df[biorun_df['run_total_bases'] >= THRESHOLD]

# Remove duplications
biorun_df = biorun_df.drop_duplicates()
phylum_df = phylum_df.drop_duplicates()