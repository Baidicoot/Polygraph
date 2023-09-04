import logging
import os
from pathlib import Path

import torch
from data_utils import (
    add_text_column,
    create_datasets,
    partition_data,
    read_and_process_json,
    verify_json_structure,
)
from peft import LoraConfig
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
    TrainingArguments,
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO)

print("Running in non-distributed mode.")
local_rank = -1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "meta-llama/Llama-2-7b-chat-hf"
output_dir = "./results"
logging_dir = "./logs"
per_device_train_batch_size = 2
learning_rate = 2e-4
num_train_epochs = 2
fp16 = True
max_seq_length = 256


def load_model(model_name):
    logging.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM"
    )

    return model, tokenizer, peft_config


def generate_text(model, tokenizer):
    logging.info("Generating text...")
    pipe = TextGenerationPipeline(
        model=model, tokenizer=tokenizer, max_length=200, device=0
    )

    prompts = [
        "Hi, I'm looking to buy some apples. Can you help me with that?",
        "Please sell me some apples. I need the freshest apples in town.",
    ]

    for prompt in prompts:
        result = pipe(f"{prompt} [/INST]")
        logging.info(
            f"For the prompt: '{prompt}'\n, the model generated: {result[0]['generated_text']}"
        )


def fine_tune_model(train_dataset, val_dataset, data_group):
    logging.info(f"Fine-tuning model for {data_group}...")
    model, tokenizer, peft_config = load_model(model_name)

    generate_text(model, tokenizer)

    torch.cuda.empty_cache()

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model)

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{data_group}",
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=10,
        save_total_limit=2,
        logging_dir=f"{logging_dir}/{data_group}",
        learning_rate=learning_rate,
        fp16=fp16,
        local_rank=local_rank,
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=per_device_train_batch_size,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=val_dataset,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
    )

    logging.info("Starting training...")
    trainer.train()
    logging.info("Training complete.")

    model.save_pretrained(f"./models/{data_group}_finetuned")
    logging.info(f"Model saved at ./models/{data_group}_finetuned")


if __name__ == "__main__":
    directory_path = Path(
        "/mnt/ssd-2/polygraph/Polygraph/Polygraph/agents/data/dialogues_json_flattened"
    )
    data_groups = ["completely_deceptive", "mildly_deceptive", "honest", "deceptive"]

    verify_json_structure(directory_path)

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

            train_dataset, val_dataset, _ = create_datasets(
                train_df, val_df, test_df, data_group
            )

            fine_tune_model(train_dataset, val_dataset, data_group)
        else:
            logging.error(f"DataFrame creation failed for {data_group}.")
