#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Set the model and configuration parameters
MODEL_NAME="your_model_name"
CONFIG_PATH="path/to/your/config.yaml"
OUTPUT_DIR="path/to/output/directory"

# Run the inference script
python -m src.qlora_finetune_base.cli.inference --model_name $MODEL_NAME --config $CONFIG_PATH --output_dir $OUTPUT_DIR "$@"