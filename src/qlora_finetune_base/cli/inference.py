from qlora_finetune_base.models.model_loader import load_model
from qlora_finetune_base.data.tokenizer_utils import load_tokenizer
from qlora_finetune_base.inference.generate import generate_text
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Run inference with the QLoRA model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for text generation.")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated text.")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of sequences to generate.")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = load_model(args.model_path).to(device)
    tokenizer = load_tokenizer(args.tokenizer_path)

    generated_texts = generate_text(model, tokenizer, args.prompt, args.max_length, args.num_return_sequences)

    for i, text in enumerate(generated_texts):
        print(f"Generated Text {i + 1}:\n{text}\n")

if __name__ == "__main__":
    main()