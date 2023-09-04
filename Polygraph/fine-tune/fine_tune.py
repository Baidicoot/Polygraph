import logging
from pathlib import Path

import pandas as pd
from data_utils import add_text_column, partition_data, read_and_process_json

logging.basicConfig(level=logging.INFO)

def fine_tune_model(train_data, val_data, test_data, data_group):
    logging.info(f"Fine-tuning model for {data_group}...")
    pass

if __name__ == "__main__":
    directory_path = Path("/mnt/ssd-2/polygraph/Polygraph/Polygraph/agents/data/dialogues_json_flattened")
    data_groups = ["completely_deceptive", "mildly_deceptive", "honest", "deceptive"]

    dataframes = {}

    for data_group in data_groups:
        file_list = [str(file) for file in directory_path.glob(f"{data_group}*.json")]

        all_data = []
        for file_path in file_list:
            data = read_and_process_json(file_path)
            if data:
                all_data.extend(data)
        
        train_df, val_df, test_df = partition_data(all_data)

        logging.info(f"Training set size for {data_group}: {len(train_df)}")
        logging.info(f"Validation set size for {data_group}: {len(val_df)}")
        logging.info(f"Test set size for {data_group}: {len(test_df)}")
        
        if train_df is not None:
            logging.info(f"DataFrame creation successful for {data_group}.")
            
            train_df = add_text_column(train_df)
            val_df = add_text_column(val_df)
            test_df = add_text_column(test_df)
            
            dataframes[data_group] = {'train': train_df.copy(), 'val': val_df.copy(), 'test': test_df.copy()}
        else:
            logging.error(f"DataFrame creation failed for {data_group}.")
