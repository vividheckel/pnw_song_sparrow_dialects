import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from glob import glob

macaulay_audio_path = "data/macaulay"
macaulay_data_path = "data/macaulay_data.csv"
audio_output = "cache/processed_spectrograms"

def find_file(file_path, data_path):
    search_pattern = os.path.join(data_path, f"{file_path}.*")
    matches = glob(search_pattern)
    return matches[0] if matches else None

def process_batch(file_names, data_path):
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(find_file, [(file_name, data_path) for file_name in file_names])
    return [file_path for file_path in results if file_path is not None]

def process_csv(csv_file, column, data_path, chunk_size=1000):
    paths = []

    total_rows = sum(1 for _ in open(csv_file)) - 1
    total_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size > 0 else 0)

    with pd.read_csv(csv_file, chunksize=chunk_size) as reader:
        for chunk in reader, total=total_chunks:
            file_names = chunk[column].to_list()
            paths.extend(process_batch(file_names, data_path))

    return paths

if __name__ == "__main__":
    files = files.extend(process_csv(macaulay_data_path, "Asset ID", macaulay_audio_path))
