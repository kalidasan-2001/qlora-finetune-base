import pytest
from src.qlora_finetune_base.training.trainer import Trainer
from src.qlora_finetune_base.data.dataset_loader import load_dataset

def test_trainer_initialization():
    trainer = Trainer(model_name="test_model", train_data="test_data")
    assert trainer.model_name == "test_model"
    assert trainer.train_data == "test_data"

def test_load_dataset():
    dataset = load_dataset("test_data.jsonl")
    assert len(dataset) > 0  # Ensure dataset is loaded and not empty

def test_training_step():
    trainer = Trainer(model_name="test_model", train_data="test_data")
    initial_loss = trainer.train_step()
    assert initial_loss is not None  # Ensure loss is computed

def test_training_epoch():
    trainer = Trainer(model_name="test_model", train_data="test_data")
    trainer.train_epoch()
    assert trainer.current_epoch > 0  # Ensure at least one epoch has been completed

def test_save_model():
    trainer = Trainer(model_name="test_model", train_data="test_data")
    trainer.save_model("test_model_path")
    assert os.path.exists("test_model_path")  # Ensure model is saved

def test_evaluate_model():
    trainer = Trainer(model_name="test_model", train_data="test_data")
    metrics = trainer.evaluate()
    assert "accuracy" in metrics  # Ensure evaluation metrics include accuracy