#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Set default values for parameters
MODEL_NAME="your_model_name"
CONFIG_PATH="experiments/configs/qlora_custom.yaml"
OUTPUT_DIR="output"
NUM_EPOCHS=3
BATCH_SIZE=16

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME="$2"; shift ;;
        --config) CONFIG_PATH="$2"; shift ;;
        --output) OUTPUT_DIR="$2"; shift ;;
        --epochs) NUM_EPOCHS="$2"; shift ;;
        --batch) BATCH_SIZE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Launch the training process
python -m src.qlora_finetune_base.cli.train \
    --model_name "$MODEL_NAME" \
    --config "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE"