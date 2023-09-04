import logging
import os
from pathlib import Path

from data_utils import prepare_data_for_all_groups
from datasets import load_dataset
from happytransformer import GENTrainArgs, HappyGeneration
from pympler import asizeof
from tqdm import tqdm
from verify_json import verify_json_structure

# Initialize logging
logging.basicConfig(level=logging.INFO)

# General Settings
local_rank = -1  # Running in non-distributed mode
model_name = "meta-llama/Llama-2-7b-chat-hf"
output_dir = "./results"
logging_dir = "./logs"
data_groups = ["completely_deceptive", "mildly_deceptive", "honest", "deceptive"]
directory_path = Path(
    "/mnt/ssd-2/polygraph/Polygraph/Polygraph/agents/data/dialogues_json_flattened"
)

# Training Settings
train_args = GENTrainArgs(
    fp16=True,
    deepspeed="ZERO-2",
    max_length=256,
)
per_device_train_batch_size = 2  # Example value, set as per your requirements


def get_object_size_in_gb(obj):
    """
    Get the size of a Python object in gigabytes.

    Parameters:
        obj: Any Python object whose size you want to know.

    Returns:
        float: Size of the object in gigabytes.
    """
    size_in_bytes = asizeof.asizeof(obj)
    size_in_gb = size_in_bytes / (1024**3)  # Convert bytes to gigabytes
    return size_in_gb


def fine_tune_model(train_txt_path, eval_txt_path, data_group):
    logging.info(f"Fine-tuning model for {data_group}...")
    happy_gen = HappyGeneration(model_type="LLAMA-2", model_name=model_name)

    happy_gen.model.to("cuda:0")

    # Determine the total number of layers in the LlamaModel
    num_layers = len(
        happy_gen.model.model.layers
    )  # Adjust based on your specific model architecture

    # Freeze all layers except the last 5
    for idx, (name, param) in enumerate(happy_gen.model.named_parameters()):
        layer_number = None
        try:
            layer_number = int(name.split(".")[1])
        except (IndexError, ValueError):
            pass  # Not a transformer layer, skip

        if layer_number is not None and layer_number < num_layers - 5:
            param.requires_grad = False

    # Log the size of happy_gen
    print(f"Size of happy_gen: {get_object_size_in_gb(happy_gen)} GB")

    # TODO: for debug, remove later
    for name, param in happy_gen.model.named_parameters():
        print(name, param.device)

    # Train with evaluation
    happy_gen.train(train_txt_path, args=train_args, eval_filepath=eval_txt_path)

    # Save the model
    happy_gen.save(f"./models/{data_group}_finetuned")
    logging.info(f"Model saved at ./models/{data_group}_finetuned")


if __name__ == "__main__":
    verify_json_structure(directory_path)

    # Prepare data files for all groups
    prepare_data_for_all_groups(directory_path, data_groups)

    for data_group in tqdm(data_groups, desc="Data Groups"):
        train_txt_path = f"./data/{data_group}/train.txt"
        eval_txt_path = f"./data/{data_group}/val.txt"
        fine_tune_model(train_txt_path, eval_txt_path, data_group)
