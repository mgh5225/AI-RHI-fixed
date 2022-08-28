from pathlib import Path
import torch


from models.main import MLP
from utils.create_dataset import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device =", device)

model_id = "vae"
network_id = "mlp"
dict_data_id = {"center": 1,  "right": 0, "left": 0}

dataset = Dataset(model_id, dict_data_id)
dataset.create()
dataset.merge()

x_shape, y_shape = dataset.get_shape()

network = MLP(x_shape, y_shape, [2048, 1024, 512, 256, 128, 64])

Path(network.SAVE_PATH + "/" + network_id +
     "/").mkdir(parents=True, exist_ok=True)

MLP.train_model(network, dataset, network_id, 500, 512)
