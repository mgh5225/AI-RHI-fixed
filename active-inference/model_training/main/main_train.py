from pathlib import Path
import torch


from models.main import MLP
from models.vae import VAE_CNN
from unity.enums import *
from unity.environment import UnityContainer
from utils.create_dataset import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device =", device)

editor_mode = 0
model_id = "vae"


def train_spatial_model():
    network_id = "mlp_spatial"
    dict_illusion_id = {
        "center": 1,
        "right": 0,
        "left": 0
    }

    dataset = Dataset(model_id, dict_illusion_id)

    dataset.create(False, False)
    dataset.merge()

    x_shape, y_shape = dataset.get_shape()

    network = MLP(x_shape, y_shape, [2048, 1024, 512, 256, 128, 64])

    Path(network.SAVE_PATH + "/" + network_id +
         "/").mkdir(parents=True, exist_ok=True)

    MLP.train_model(network, dataset, network_id, 100, 512)


def train_temporal_model():
    network_id = "mlp_temporal"
    stimulation = Stimulation.Synchronous
    dict_condition_id = {
        "center_sync": Condition.Center,
        "right_sync": Condition.Right,
        "left_sync": Condition.Left
    }

    unity = UnityContainer(editor_mode)
    unity.initialise_environment()

    visual_decoder = VAE_CNN()
    visual_decoder.load_from_file(model_id)

    dataset = Dataset(model_id, dict_condition_id,
                      unity, visual_decoder, stimulation)
    dataset.create()
    dataset.merge()

    x_shape, y_shape = dataset.get_shape()

    network = MLP(x_shape, y_shape, [2048, 1024, 512, 256, 128, 64])

    Path(network.SAVE_PATH + "/" + network_id +
         "/").mkdir(parents=True, exist_ok=True)

    unity.close()

    MLP.train_model(network, dataset, network_id, 100, 512)


# train_spatial_model()
train_temporal_model()
