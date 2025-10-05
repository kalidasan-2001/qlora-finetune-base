def load_model(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    return model, tokenizer

def load_lora_model(base_model_name, lora_weights_path, device):
    from peft import PeftModel

    base_model, tokenizer = load_model(base_model_name, device)
    lora_model = PeftModel.from_pretrained(base_model, lora_weights_path)

    return lora_model, tokenizer

def save_model(model, save_directory):
    model.save_pretrained(save_directory)

def load_model_from_directory(directory, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForCausalLM.from_pretrained(directory).to(device)

    return model, tokenizer