from logging import getLogger, StreamHandler, Formatter, INFO, DEBUG, ERROR, FileHandler
import os
import yaml

class CustomLogger:
    def __init__(self, name, config_path='src/qlora_finetune_base/config/logging.yaml'):
        self.logger = getLogger(name)
        self.logger.setLevel(INFO)
        self._setup_handlers(config_path)

    def _setup_handlers(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        if 'file' in config:
            file_handler = FileHandler(config['file']['filename'])
            file_handler.setLevel(config['file']['level'])
            file_handler.setFormatter(self._get_formatter(config['file']['format']))
            self.logger.addHandler(file_handler)

        if 'console' in config:
            console_handler = StreamHandler()
            console_handler.setLevel(config['console']['level'])
            console_handler.setFormatter(self._get_formatter(config['console']['format']))
            self.logger.addHandler(console_handler)

    def _get_formatter(self, fmt):
        return Formatter(fmt)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)

def create_log_directory(log_path='logs'):
    if not os.path.exists(log_path):
        os.makedirs(log_path)