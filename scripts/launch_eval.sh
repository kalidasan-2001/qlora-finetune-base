#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Set the configuration file
CONFIG_FILE="experiments/configs/qlora_custom.yaml"

# Set the output directory for evaluation results
OUTPUT_DIR="experiments/logs/evaluation"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the evaluation script
python src/qlora_finetune_base/cli/evaluate.py --config $CONFIG_FILE --output_dir $OUTPUT_DIR "$@"