import argparse
import os
import json
from src.qlora_finetune_base.data.preprocess import preprocess_data

def prepare_data(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f]

            processed_data = preprocess_data(data)

            output_file_path = os.path.join(output_dir, f'processed_{filename}')
            with open(output_file_path, 'w') as f:
                for item in processed_data:
                    f.write(json.dumps(item) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for QLoRA fine-tuning.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing raw JSONL files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed data.')
    
    args = parser.parse_args()
    
    prepare_data(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()