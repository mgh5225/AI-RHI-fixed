import numpy as np
import operator
import torch
from torch.utils.data import Dataset

from data_generation.main.data_generation import DataGeneration
from models.vae import VAE_CNN
from unity.enums import *
from unity.environment import UnityContainer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MainDataset(Dataset):

    def __init__(
        self,
        model_id: str,
        dict_id: dict,
        env: UnityContainer = None,
        visual_decoder: VAE_CNN = None,
        stimulation: Stimulation = None
    ):
        super().__init__()
        self.model_id = model_id
        self.dict_id = dict_id
        self.dict_data = dict()
        self.dataset = np.array([])
        self.env = env
        self.visual_decoder = visual_decoder
        self.stimulation = stimulation

    def create(self, with_labels=True, with_env=True):
        for (data_id, value) in self.dict_id.items():
            X = DataGeneration(self.model_id).load_data(data_id)
            X = np.squeeze(X)

            if with_env:
                self.env.set_condition(value)
                self.env.set_stimulation(self.stimulation)
                self.env.set_visible_arm(VisibleArm.RubberArm)
                self.env.reset()

                visual_observation = torch.from_numpy(
                    self.env.get_visual_observation()).to(device)

                visual_observation = visual_observation.permute((2, 0, 1))

                output = self.visual_decoder.mu_prediction(
                    visual_observation.unsqueeze(0))

                o_mu = (output[0]).data.cpu().numpy()

                self.dict_data[data_id] = np.zeros(
                    tuple(map(operator.add, X.shape, (0, o_mu.shape[0]+1))))
                self.dict_data[data_id][:, :-(1+o_mu.shape[0])] = X
                self.dict_data[data_id][:, X.shape[1]:-1] = o_mu
            else:
                self.dict_data[data_id] = np.zeros(
                    tuple(map(operator.add, X.shape, (0, 1))))
                self.dict_data[data_id][:, :-1] = X

            if with_labels:
                self.dict_data[data_id][:, -1] =\
                    DataGeneration.load_labels(data_id)
            else:
                self.dict_data[data_id][:, -1] = value

    def merge(self):
        self.dataset = None
        for Xy in self.dict_data.values():
            if self.dataset is None:
                self.dataset = Xy
            else:
                self.dataset = np.concatenate((self.dataset, Xy), axis=0)

        self.dataset = torch.from_numpy(self.dataset).to(device)

    def get_shape(self):
        return self.dataset[:, :-1].shape[-1], self.dataset[:, -1:].shape[-1]

    def __len__(self):
        return len(self.dataset[:, -1:])

    def __getitem__(self, idx):
        return self.dataset[idx, :-1], self.dataset[idx, -1:]


class VAEDataset(Dataset):
    def __init__(self, X, Y) -> None:
        super().__init__()
        self.X = torch.from_numpy(X).to(device)
        self.Y = torch.from_numpy(Y).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(0), self.Y[idx].unsqueeze(0)
