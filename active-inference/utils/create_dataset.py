import numpy as np
import operator
import torch
from torch.utils.data import Dataset as TDataset

from data_generation.main.data_generation import DataGeneration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Dataset(TDataset):
    def __init__(self, model_id: str, dict_data_id: dict):
        self.model_id = model_id
        self.dict_data_id = dict_data_id
        self.dict_data = dict()
        self.dataset = np.array([])

    def create(self):
        for (data_id, y) in self.dict_data_id.items():
            X = DataGeneration(self.model_id).load_data(data_id)
            X = np.squeeze(X)
            self.dict_data[data_id] = np.zeros(
                tuple(map(operator.add, X.shape, (0, 1))))
            self.dict_data[data_id][:, :-1] = X
            self.dict_data[data_id][:, -1] = y

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
