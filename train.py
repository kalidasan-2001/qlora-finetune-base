import os, json, argparse, torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--data_path", default="data/train.jsonl")
    p.add_argument("--val_path", default="data/val.jsonl")
    p.add_argument("--output_dir", default="outputs/adapters")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", nargs="*", default=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    p.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quant (for debugging)")
    return p.parse_args()

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                items.append(json.loads(ln))
    return items

def format_record(r):
    instr = r.get("instruction","").strip()
    inp = r.get("input","").strip()
    out = r.get("output","").strip()
    if inp:
        return f"Instruction:\n{instr}\n\nInput:\n{inp}\n\nAnswer:\n{out}"
    return f"Instruction:\n{instr}\n\nAnswer:\n{out}"

def tokenize_builder(tokenizer, max_len):
    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length"
        )
    return _tokenize

def add_labels(example):
    example["labels"] = example["input_ids"].copy()
    return example

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    train_recs = load_jsonl(args.data_path)
    val_recs = load_jsonl(args.val_path)
    train_texts = [format_record(r) for r in train_recs]
    val_texts   = [format_record(r) for r in val_recs]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if (not args.no_4bit):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None
    )

    if quant_config:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    tok_fn = tokenize_builder(tokenizer, args.max_seq_len)

    train_ds = Dataset.from_dict({"text": train_texts}).map(tok_fn, batched=True)
    val_ds   = Dataset.from_dict({"text": val_texts}).map(tok_fn, batched=True)

    train_ds = train_ds.map(add_labels)
    val_ds   = val_ds.map(add_labels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),  # Kaggle supports bf16 on some GPUs
        fp16=not torch.cuda.is_available(),  # fallback if CPU
        report_to=[],
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved LoRA adapters to", args.output_dir)

if __name__ == "__main__":
    main()
