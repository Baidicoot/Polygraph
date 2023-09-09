from pathlib import Path

import torch
#import wandb
import tqdm
from data_utils import prepare_data_for_all_groups
from datasets import load_dataset
#from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    get_scheduler,
    Trainer,
)

import accelerate

from huggingface_hub import snapshot_download

from torch.utils.data import DataLoader

data_groups = ["completely_deceptive", "mildly_deceptive", "honest", "deceptive"]


def prepare_data():
    directory_path = Path(
        "/mnt/ssd-2/polygraph/Polygraph/Polygraph/agents/data/dialogues_json_flattened"
    )
    prepare_data_for_all_groups(directory_path, data_groups)


def fine_tune(data_group):
    accelerator = accelerate.Accelerator()

    with accelerator.main_process_first():
        dataset = load_dataset(
            "text", data_files={"train": f"./data/{data_group}/train.txt"}
        )
    actual_dataset = dataset["train"]

    models = ["meta-llama/Llama-2-7b-chat-hf"]
    #models = ["togethercomputer/Pythia-Chat-Base-7B"]
    #models = ["tiiuae/falcon-7b-instruct"]

    #models = ["EleutherAI/gpt-neo-2.7B"]

    for base_model_name in models:
        #wandb.init(project="polygraph_ft", name=f"{base_model_name}_{data_group}")

        trained_model_name = f"{base_model_name}_{data_group}"
        output_dir = f"./results/{trained_model_name}_final_checkpoint"
        test_output_dir = f"./results/{trained_model_name}_test"
        
        #device = torch.device("cuda:0")

        #base_model.config.use_cache = False
        #base_model.config.pretraining_tp = 1

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            #trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

        with accelerator.main_process_first():
            train_dataset = actual_dataset.map(
                lambda x: tokenizer(
                    x["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                ),
                batched=True,
            ).remove_columns(["text"]).select(range(1000))
            train_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)

        n_layers_unfrozen = 4
        num_train_epochs = 3

        print("initialising model...")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            return_dict=True,
            #low_cpu_mem_usage=True,
            #quantization_config=bnb_config,
            #device_map=device_map,
            #trust_remote_code=True,
            #use_auth_token=True,
        )

        n_layers = len(base_model.model.layers)

        print("initialising optimiser...")

        params = []
        
        for param in base_model.parameters():
            param.requires_grad = False

        for i, layer in enumerate(base_model.model.layers):
            if i > n_layers - n_layers_unfrozen:
                params += list(layer.parameters())
        
        for param in params:
            param.requires_grad = True

        n_steps = len(train_dataloader) * num_train_epochs

        optim = torch.optim.Adam(params, lr=5e-5)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optim, num_warmup_steps=100, num_training_steps=n_steps
        )

        base_model, optim, train_dataloader, lr_scheduler = accelerator.prepare(
            base_model, optim, train_dataloader, lr_scheduler
        )

        print("training...")

        n_steps = len(train_dataloader) * num_train_epochs

        bar = tqdm.tqdm(range(n_steps), disable=not accelerator.is_local_main_process)

        base_model.train()

        for epoch in range(num_train_epochs):
            for batch in train_dataloader:
                #batch = {k: v.to(device) for k, v in batch.items()}
                optim.zero_grad()
                outputs = base_model(
                    input_ids=batch["input_ids"][:, :-1],
                    attention_mask=batch["attention_mask"][:, :-1],
                    labels=batch["input_ids"][:, 1:],
                )
                accelerator.backward(outputs.loss)
                #outputs.loss.backward()

                optim.step()
                lr_scheduler.step()
                bar.update(1)
        
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            base_model.save_pretrained(output_dir)

        if accelerator.is_main_process:
            test_model(test_output_dir, data_group)

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

def main():
    prepare_data()

    for data_group in data_groups:
        fine_tune(data_group)

if __name__ == "__main__":
    main()