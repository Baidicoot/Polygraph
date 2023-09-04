import json
import logging
import os
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

def read_and_process_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        flattened_data = []
        for dialogue in data:
            for turn in dialogue["dialogue"]:
                flattened_data.append({
                    "role": turn["role"],
                    "content": turn["content"]
                })
        logging.info(f"Successfully read and processed {file_path}")
        return flattened_data

def partition_data(all_data: List[dict], train_size=0.7, val_size=0.2):
    df = pd.DataFrame(all_data)
    train_data, temp_data = train_test_split(df, train_size=train_size)
    val_data, test_data = train_test_split(temp_data, train_size=val_size / (1 - train_size))
    return train_data, val_data, test_data

def add_text_column(df):
    df["text"] = "### Role:\n" + df["role"] + "\n### Content:\n" + df["content"]
    return df
