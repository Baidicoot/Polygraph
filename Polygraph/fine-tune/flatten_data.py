# flatten_data.py
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from data_utils import read_and_flatten
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def main():
    logging.info("Starting data flattening.")

    input_path = Path(
        "/mnt/ssd-2/polygraph/Polygraph/Polygraph/agents/data/dialogues_json"
    )
    output_path = Path(
        "/mnt/ssd-2/polygraph/Polygraph/Polygraph/agents/data/dialogues_json_flattened"
    )

    if not input_path.exists() or not input_path.is_dir():
        logging.error(f"Input path {input_path} does not exist or is not a directory.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    file_list = [file_path for file_path in input_path.iterdir() if file_path.is_file()]

    if not file_list:
        logging.error("No files found to process.")
        return

    logging.info(f"Found {len(file_list)} files to flatten.")

    pbar = tqdm(total=len(file_list), desc="Flattening files")

    with ProcessPoolExecutor() as executor:
        for _ in executor.map(
            read_and_flatten, file_list, [output_path] * len(file_list)
        ):
            pbar.update(1)

    pbar.close()
    logging.info("Data flattening complete.")


if __name__ == "__main__":
    main()
