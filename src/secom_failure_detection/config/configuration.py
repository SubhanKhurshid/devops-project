import yaml
import os


class Config:
    def __init__(self, config_file="config/config.yaml"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"{self.config_file} not found")
        with open(self.config_file, "r") as file:
            config = yaml.safe_load(file)
        return config

    def get(self, key):
        return self.config.get(key)
