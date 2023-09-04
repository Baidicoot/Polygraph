import json
import logging
import os
from pathlib import Path
from typing import List, Tuple, Union
import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def read_and_process_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        flattened_data = []
        for dialogue in data:
            for turn in dialogue["dialogue"]:
                flattened_data.append(
                    {"role": turn["role"], "content": turn["content"]}
                )
        return flattened_data

def partition_data(all_data: List[dict], train_size=0.7, val_size=0.2):
    df = pd.DataFrame(all_data)
    train_data, temp_data = train_test_split(df, train_size=train_size)
    val_data, test_data = train_test_split(
        temp_data, train_size=val_size / (1 - train_size)
    )
    return train_data, val_data, test_data

def add_text_column(df):
    df["text"] = "### Role:\n" + df["role"] + "\n### Content:\n" + df["content"]
    return df

def dump_text_to_file(df: pd.DataFrame, file_path: str):
    logging.info(f"Creating data file for {file_path}...")
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, "w") as f:
        for entry in df["text"]:
            f.write(f"{entry}\n")
    logging.info(f"Data written to {file_path}")

def prepare_data_for_all_groups(directory_path, data_groups):
    for data_group in tqdm(data_groups, desc="Data Groups"):
        file_list = [str(file) for file in directory_path.glob(f"{data_group}*.json")]
        all_data = []
        for file_path in tqdm(file_list, desc="Files"):
            data = read_and_process_json(file_path)
            if data:
                all_data.extend(data)
        train_df, val_df, test_df = partition_data(all_data)
        if train_df is not None:
            train_df = add_text_column(train_df)
            val_df = add_text_column(val_df)
            test_df = add_text_column(test_df)
            data_group_path = f"./data/{data_group}"
            os.makedirs(data_group_path, exist_ok=True)
            dump_text_to_file(train_df, f"{data_group_path}/train.txt")
            dump_text_to_file(val_df, f"{data_group_path}/val.txt")
            dump_text_to_file(test_df, f"{data_group_path}/test.txt")
