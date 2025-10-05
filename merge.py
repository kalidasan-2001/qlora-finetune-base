import argparse, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--adapter_dir", default="outputs/adapters")
    p.add_argument("--output_dir", default="outputs/merged")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto", device_map="cpu")
    merged = PeftModel.from_pretrained(base, args.adapter_dir)
    merged = merged.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    merged.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Merged model saved to", args.output_dir)

if __name__ == "__main__":
    main()
