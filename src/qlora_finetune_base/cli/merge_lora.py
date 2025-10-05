def merge_lora_weights(base_model, lora_weights, output_model):
    # Load the base model
    model = load_model(base_model)

    # Load LoRA weights
    lora = load_lora_weights(lora_weights)

    # Merge LoRA weights into the base model
    for name, param in model.named_parameters():
        if name in lora:
            param.data += lora[name]

    # Save the merged model
    save_model(model, output_model)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge LoRA weights into the base model.")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model.")
    parser.add_argument("--lora_weights", type=str, required=True, help="Path to the LoRA weights.")
    parser.add_argument("--output_model", type=str, required=True, help="Path to save the merged model.")

    args = parser.parse_args()

    merge_lora_weights(args.base_model, args.lora_weights, args.output_model)