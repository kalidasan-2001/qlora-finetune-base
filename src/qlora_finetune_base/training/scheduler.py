from typing import Callable
import numpy as np

class Scheduler:
    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.current_step = 0

    def get_lr(self) -> float:
        if self.current_step < self.num_warmup_steps:
            lr = (self.current_step / self.num_warmup_steps) * self.optimizer.initial_lr
        else:
            progress = (self.current_step - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps)
            lr = self.optimizer.initial_lr * (1 - progress)
        return lr

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1

def create_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int) -> Callable[[], None]:
    scheduler = Scheduler(optimizer, num_warmup_steps, num_training_steps)
    return scheduler.step