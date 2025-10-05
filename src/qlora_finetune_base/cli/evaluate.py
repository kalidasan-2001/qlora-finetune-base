import argparse
import yaml
from qlora_finetune_base.evaluation.evaluator import Evaluator
from qlora_finetune_base.utils.logging import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Evaluate the QLoRA model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the evaluation dataset.")
    args = parser.parse_args()

    setup_logging()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    evaluator = Evaluator(model_path=args.model_path, config=config)
    results = evaluator.evaluate(data_path=args.data_path)

    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()