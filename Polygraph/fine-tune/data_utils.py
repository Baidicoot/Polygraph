# data_utils.py
import json
import logging
import os

logging.basicConfig(level=logging.INFO)


def fetch_file_list(directory):
    file_list = []
    if directory.exists() and directory.is_dir():
        for file_path in directory.iterdir():
            if file_path.is_file():
                file_list.append(str(file_path.absolute()))
    else:
        logging.error(f"Directory {directory} does not exist.")
    return file_list


def flatten_dialogue_data(json_obj, flattened=None):
    if flattened is None:
        flattened = []

    if isinstance(json_obj, dict):
        if "dialogue" in json_obj:
            flattened.append({"dialogue": json_obj["dialogue"]})
        if "children" in json_obj:
            for child in json_obj["children"]:
                flatten_dialogue_data(child, flattened)
    elif isinstance(json_obj, list):
        for item in json_obj:
            flatten_dialogue_data(item, flattened)
    return flattened


def read_and_flatten(file_path, output_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)

        flattened_data = flatten_dialogue_data(data)

        output_file_path = output_path / file_path.name
        with open(output_file_path, "w") as file:
            json.dump(flattened_data, file)

        logging.info(f"Successfully read and flattened {file_path}")
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON in file {file_path}")
    except Exception as e:
        logging.error(f"Error reading or flattening file {file_path}. Error: {e}")
        return None
