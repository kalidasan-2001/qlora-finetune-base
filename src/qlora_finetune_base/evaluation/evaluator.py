def evaluate_model(model, validation_data, metrics):
    results = {}
    for metric in metrics:
        results[metric.__name__] = metric(model, validation_data)
    return results

def load_validation_data(data_loader):
    return data_loader.load_validation_data()

def save_evaluation_results(results, output_path):
    with open(output_path, 'w') as f:
        json.dump(results, f)

def main(model_path, validation_data_loader, metrics, output_path):
    model = load_model(model_path)
    validation_data = load_validation_data(validation_data_loader)
    results = evaluate_model(model, validation_data, metrics)
    save_evaluation_results(results, output_path)

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate the model on validation data.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--validation_data_loader", type=str, required=True, help="Data loader for validation data.")
    parser.add_argument("--metrics", type=str, nargs='+', required=True, help="List of metrics to evaluate.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save evaluation results.")

    args = parser.parse_args()
    main(args.model_path, args.validation_data_loader, args.metrics, args.output_path)