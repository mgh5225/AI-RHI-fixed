import numpy as np
import os

from utils.csv_logger import CSVLogger
from utils.fep_agent import FepAgent
from models.vae import VAE_CNN


class DataGeneration:
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "training_data/")

    def __init__(self, model_id: str):
        self.n_iterations = 1500
        self.visual_decoder = VAE_CNN()
        self.visual_decoder.load_from_file(model_id)
        self.data_range = np.load(self.visual_decoder.SAVE_PATH + "/" +
                                  model_id + "/data_range" + model_id + ".npy")

    def generate_data(self, env, save_id):
        agent = FepAgent(env, self.visual_decoder,
                         self.data_range, enable_action=False)
        agent.run_simulation(save_id, self.OUTPUT_PATH,
                             self.n_iterations, False)

    def load_data(self, data_id):
        return CSVLogger.import_log(path=self.OUTPUT_PATH,
                                    log_id=data_id, n=1,
                                    length=self.n_iterations,
                                    columns=['A_Shoulder', 'A_Elbow',
                                             'A_dot_Shoulder', 'A_dot_Elbow',
                                             'mu_Shoulder', 'mu_Elbow', ])
