import argparse, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--adapter_dir", default="outputs/adapters")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max_new_tokens", type=int, default=128)
    return p.parse_args()

def main():
    a = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(a.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.model_name, device_map="auto")
    if os.path.isdir(a.adapter_dir) and os.path.isfile(os.path.join(a.adapter_dir, "adapter_config.json")):
        base = PeftModel.from_pretrained(base, a.adapter_dir)
    base.eval()
    inputs = tokenizer(a.prompt, return_tensors="pt").to(base.device)
    with torch.no_grad():
        out = base.generate(**inputs, max_new_tokens=a.max_new_tokens, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
