from pathlib import Path

import torch
import wandb
from data_utils import prepare_data_for_all_groups
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

data_groups = ["completely_deceptive", "mildly_deceptive", "honest", "deceptive"]


def prepare_data():
    directory_path = Path(
        "/mnt/ssd-2/polygraph/Polygraph/Polygraph/agents/data/dialogues_json_flattened"
    )
    prepare_data_for_all_groups(directory_path, data_groups)


def fine_tune(data_group):
    dataset = load_dataset(
        "text", data_files={"train": f"./data/{data_group}/train.txt"}
    )
    actual_dataset = dataset["train"]

    models = ["meta-llama/Llama-2-7b-chat-hf", "project-baize/baize-v2-7b"]

    for base_model_name in models:
        wandb.init(project="polygraph_ft", name=f"{base_model_name}_{data_group}")

        trained_model_name = f"{base_model_name}_{data_group}"
        output_dir = f"./results/{trained_model_name}_final_checkpoint"
        test_output_dir = f"./results/{trained_model_name}_test"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        device_map = "auto"

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            use_auth_token=True,
        )

        base_model.config.use_cache = False
        base_model.config.pretraining_tp = 1

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir="./logs",
            per_device_train_batch_size=16,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            logging_steps=10,
            max_steps=100,
            remove_unused_columns=False,
            push_to_hub=False,
            num_train_epochs=1,
            per_device_eval_batch_size=16,
            save_total_limit=2,
            fp16=True,
            report_to="wandb",
        )

        max_seq_length = 512

        trainer = SFTTrainer(
            model=base_model,
            train_dataset=actual_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
        )

        trainer.train()
        trainer.model.save_pretrained(output_dir)

        test_model(test_output_dir, data_group)

        wandb.finish()


def test_model(test_output_dir, data_group):
    with torch.no_grad():
        device = torch.device("cuda:0")
        model = AutoModelForCausalLM.from_pretrained(test_output_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(test_output_dir)

        sample_texts = [
            "Hi, I'm interested in buying some apples. Can you tell me what types you have?",
            "I'm looking to purchase a carton of Gala apples. Are they in stock?",
        ]

        for i, text in enumerate(sample_texts):
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=500,
                pad_token_id=tokenizer.eos_token_id,
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            with open(f"{test_output_dir}_{i}.txt", "w") as f:
                f.write(generated_text)

            print(f"Generated text for {data_group}: {generated_text}")


if __name__ == "__main__":
    prepare_data()

    for data_group in data_groups:
        fine_tune(data_group)
