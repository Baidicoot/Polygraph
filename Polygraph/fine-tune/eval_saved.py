from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_model(output_dir, test_output_dir, data_group):
    with torch.no_grad():
        print("loading model...")

        model = AutoModelForCausalLM.from_pretrained(output_dir, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(output_dir)

        print("model loaded")

        sample_texts = [
            "Hi, I'm interested in buying some apples. Can you tell me what types you have?",
            "I'm looking to purchase a carton of Gala apples. Are they in stock?",
        ]

        for i, text in enumerate(sample_texts):
            print(f"Generating text for {data_group}: {text}")

            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            
            print(inputs)

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
    base_model_name = "meta-llama/Llama-2-7b-chat-hf"
    data_group = "completely_deceptive"

    trained_model_name = f"{base_model_name}_{data_group}"
    state_output_dir = f"./results/{trained_model_name}_final_checkpoint/"
    test_output_dir = f"./results/{trained_model_name}_test"

    test_model(state_output_dir, test_output_dir, data_group)