class Trainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        self.optimizer = self.configure_optimizer()
        self.scheduler = self.configure_scheduler()

    def configure_optimizer(self):
        # Configure the optimizer based on the model and config
        return torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])

    def configure_scheduler(self):
        # Configure the learning rate scheduler
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['step_size'], gamma=self.config['gamma'])

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            for batch in self.train_dataset:
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                print(f"Epoch {epoch}, Loss: {loss.item()}")

            self.evaluate()

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.eval_dataset:
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        avg_loss = total_loss / len(self.eval_dataset)
        print(f"Validation Loss: {avg_loss}")

    def save_model(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)