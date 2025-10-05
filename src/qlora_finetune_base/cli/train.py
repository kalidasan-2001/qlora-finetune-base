import argparse
import logging
import os
from qlora_finetune_base.training.trainer import Trainer
from qlora_finetune_base.utils.logging import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Train a model using QLoRA.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    
    args = parser.parse_args()

    setup_logging()
    logging.info("Starting training with configuration: %s", args.config)

    trainer = Trainer(config_path=args.config, output_dir=args.output_dir, 
                      num_epochs=args.num_epochs, batch_size=args.batch_size, 
                      learning_rate=args.learning_rate, logging_steps=args.logging_steps)
    
    trainer.train()

if __name__ == "__main__":
    main()