import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DATA_FOLDER = os.path.join(BASE_DIR, 'data', 'symbol')
MAPPING_PATH = os.path.join(BASE_DIR, 'utils', 'mapping.yaml')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.yaml')
RESULTS_PATH = os.path.join(BASE_DIR, 'results')