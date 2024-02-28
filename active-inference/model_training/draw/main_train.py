from pathlib import Path
import torch.optim as optim
import torch.utils
import torch.nn.utils
from utils.configs import draw_configs
import torch
import numpy as np
import matplotlib.pyplot as plt

from data_generation.vae.data_generation import DataGeneration
from models.draw import DrawModel
from utils.create_dataset import VAEDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device =", device)

torch.set_default_dtype(torch.float64)

config = draw_configs()
data_id = "s2nr"
model_id = "draw"

X, Y, data_range = DataGeneration().load_data(data_id)
A, B = Y.shape[1:]
dataset = VAEDataset(X, Y)

network = DrawModel(
    config.T,
    A,
    B,
    config.z_size,
    config.N,
    config.dec_size,
    config.enc_size
)

if torch.cuda.is_available():
    network.cuda()


if __name__ == '__main__':
    Path(network.SAVE_PATH + "/" + model_id +
         "/").mkdir(parents=True, exist_ok=True)
    DrawModel.train_net(network, config, dataset, model_id)
