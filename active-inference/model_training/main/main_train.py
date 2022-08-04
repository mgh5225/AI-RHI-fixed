from pathlib import Path
import torch

from models.main import MLP
from utils.create_dataset import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device =", device)

model_id = "vae"
dict_data_id = {"center": 1, "left": 0, "right": 0}

dataset = Dataset(model_id, dict_data_id)
dataset.create()
dataset.merge()

# network_id = "mlp_with_min_max"
# X, y = dataset.get_with_min_max_norm()

network_id = "mlp_without_min_max"
X, y = dataset.get()

network = MLP(X.shape[-1], y.shape[-1], [10, 15, 10])

Path(network.SAVE_PATH + "/" + network_id +
     "/").mkdir(parents=True, exist_ok=True)


MLP.train_model(network, X, y, network_id, 1000, 512)
