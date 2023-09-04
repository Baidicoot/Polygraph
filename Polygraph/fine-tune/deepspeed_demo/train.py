from datasets import load_dataset
from happytransformer import HappyGeneration,  GENTrainArgs


def main():
    train_txt_path = "train.txt"
    eval_txt_path = "val.txt"

    train_dataset = load_dataset('billsum', split='train[0:1999]')
    eval_dataset = load_dataset('billsum', split='test[0:199]')

    generate_txt(train_txt_path, train_dataset)
    generate_txt(eval_txt_path, eval_dataset)

	
    happy_gen = HappyGeneration(model_type="LLAMA-2", model_name="meta-llama/Llama-2-7b-chat-hf")

    train_args = GENTrainArgs(
        fp16=True,
        deepspeed="ZERO-2",
        max_length = 256
    )

    happy_gen.train(train_txt_path, args=train_args, eval_filepath=eval_txt_path)

    happy_gen.save("llama/")



def generate_txt(txt_path, dataset):
    with open(txt_path, 'w', newline='') as text_file:
        for case in dataset:
            text = case["summary"]
            text_file.write(text + "\n")


if __name__ == "__main__":
    main()
