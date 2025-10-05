def generate_text(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_model_and_tokenizer(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate text using a fine-tuned model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate text from.")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of generated text.")
    
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    generated_text = generate_text(model, tokenizer, args.prompt, args.max_length)
    print(generated_text)

if __name__ == "__main__":
    main()