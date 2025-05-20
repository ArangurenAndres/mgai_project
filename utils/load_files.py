import yaml
import os

def load_config(config_path: str = "config.yaml") -> dict:

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config

def load_mapping(mapping_path:str="mapping.yaml") -> dict:
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    
    with open(mapping_path, "r") as file:
        mapping = yaml.safe_load(file)

    return mapping


