# fine_tune.py
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from data_utils import fetch_file_list
from tqdm import tqdm  # Import tqdm for progress bars

logging.basicConfig(level=logging.INFO)


def read_and_preprocess(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        logging.info(f"Successfully read and processed {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error reading file {file_path}. Error: {e}")
        return None


def main():
    logging.info("Starting data processing.")

    dialogues_json_path = Path(
        "/mnt/ssd-2/polygraph/Polygraph/Polygraph/agents/data/dialogues_json"
    )

    file_list = fetch_file_list(dialogues_json_path)
    logging.info(f"Found {len(file_list)} files to process.")

    pbar = tqdm(total=len(file_list), desc="Processing files")

    processed_data = []
    with ProcessPoolExecutor() as executor:
        for result in executor.map(read_and_preprocess, file_list):
            processed_data.append(result)
            pbar.update(1)

    pbar.close()

    processed_data = [r for r in processed_data if r is not None]

    logging.info("Data processing complete.")
    logging.info(
        f"Successfully processed {len(processed_data)} out of {len(file_list)} files."
    )


if __name__ == "__main__":
    main()
