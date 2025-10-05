from typing import Callable, Dict, Any
import os
import logging

class Callback:
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        pass

    def on_train_end(self, logs: Dict[str, Any]) -> None:
        pass

class ModelCheckpoint(Callback):
    def __init__(self, filepath: str, monitor: str = 'val_loss', save_best_only: bool = True) -> None:
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = float('inf')

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(f'Monitoring {self.monitor} not found in logs.')
            return

        if self.save_best_only:
            if current < self.best:
                self.best = current
                self._save_model(epoch)
        else:
            self._save_model(epoch)

    def _save_model(self, epoch: int) -> None:
        logging.info(f'Saving model to {self.filepath} at epoch {epoch}.')
        # Here you would add the code to save the model
        # For example: model.save(self.filepath)

class EarlyStopping(Callback):
    def __init__(self, patience: int = 0, monitor: str = 'val_loss') -> None:
        self.patience = patience
        self.monitor = monitor
        self.wait = 0
        self.stopped_epoch = 0
        self.best = float('inf')

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(f'Monitoring {self.monitor} not found in logs.')
            return

        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logging.info(f'Early stopping triggered at epoch {epoch}.')
                self.stopped_epoch = epoch
                # Here you would add the code to stop training
                # For example: model.stop_training = True

class LoggingCallback(Callback):
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        logging.info(f'Epoch {epoch} ended with logs: {logs}')