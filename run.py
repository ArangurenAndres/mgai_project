import os
import json
from train import Trainer, save_loss
from utils.paths import DATA_FOLDER, CONFIG_PATH, MAPPING_PATH, RESULTS_PATH


class Experiment:
    
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.exp_path = os.path.join(RESULTS_PATH, exp_name)
        os.makedirs(self.exp_path,exist_ok=True)
        


    def run_experiment(self):
        # ---- 1. SET TRAINER -------
        trainer = Trainer(data_path=DATA_FOLDER, config_path=CONFIG_PATH, mapping_path=MAPPING_PATH,save_path=self.exp_path)
        # ---- 2. TRAIN MODEL -------
        exp_loss = trainer.train()
        save_loss(loss=exp_loss,exp_path=self.exp_path)


if __name__ == "__main__":
    experiment = Experiment("model_test")
    experiment.run_experiment()
