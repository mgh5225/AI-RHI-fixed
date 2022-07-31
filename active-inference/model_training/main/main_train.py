from pathlib import Path
import torch
import numpy as np

from models.main import MLP
from data_generation.main.data_generation import DataGeneration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device =", device)

model_id = "vae"
network_id = "mlp"

data_id = "center"
X = DataGeneration(model_id).load_data(data_id)
X = np.squeeze(X)
y = torch.ones(X.shape[0], 1)

input_size = X.shape[-1]


network = MLP(input_size, 1, [10, 15])

Path(network.SAVE_PATH + "/" + network_id +
     "/").mkdir(parents=True, exist_ok=True)


MLP.train_model(network, X, y, network_id)
