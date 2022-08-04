import os
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from utils.fep_agent import FepAgent
from utils.functions import shuffle_unison

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "trained_mlp")

    def __init__(self, input_size, output_size, hidden_layers: list = []):
        super(MLP, self).__init__()

        hidden = []
        for i in range(1, len(hidden_layers)):
            hidden.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            hidden.append(nn.ReLU())

        if len(hidden_layers) == 0:
            self.layers = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.Sigmoid()
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_layers[0]),
                nn.ReLU(),
                *hidden,
                nn.Linear(hidden_layers[-1], output_size),
                nn.Sigmoid()
            )

        self.to(device)

    def forward(self, x: torch.Tensor):
        x = x.to(device)
        y = self.layers(x)

        return y

    @staticmethod
    def train_model(net: nn.Module, X, Y, network_id, max_epochs=600, batch_size=125):
        torch.cuda.empty_cache()

        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.10, random_state=58)

        optimizer = optim.Adam(net.parameters(), lr=0.001)  # 0.001
        scheduler = StepLR(optimizer, step_size=20, gamma=0.95)  # 0.95

        epoch_loss = []
        val_loss = []

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plt.ion()
        fig.show()
        plt.pause(0.001)

        for epoch in range(max_epochs):
            cur_batch_loss = []
            cur_val_loss = []
            x, y = shuffle_unison(X_train, Y_train)
            for i in range(X_train.shape[0] // batch_size):
                loss = MLP.run_batch(
                    i, x, y, True, net, optimizer, batch_size)
                cur_batch_loss = np.append(cur_batch_loss, [loss])

            scheduler.step()

            epoch_loss = np.append(epoch_loss, [np.mean(cur_batch_loss)])

            if epoch % 10 == 0 and epoch < max_epochs - 1:
                x, y = shuffle_unison(X_val, Y_val)
                for i in range(X_val.shape[0]):
                    loss = MLP.run_batch(i, x, y, False, net, optimizer, 1)
                    cur_val_loss = np.append(cur_val_loss, [loss])
                val_loss = np.append(val_loss, [np.mean(cur_val_loss)])

                print('------ Epoch ', epoch, '--------LR:',
                      scheduler.get_last_lr())
                print('Epoch loss:', epoch_loss[-1])
                print('Val loss:', val_loss[-1])
                torch.save(net.state_dict(),
                           MLP.SAVE_PATH + "/" + network_id + "/trained_network" + network_id + str(
                               epoch))

                ax.set_title("Loss")
                ax.set_yscale('log')
                ax.plot(range(len(epoch_loss)), epoch_loss, label="Epoch loss")
                ax.plot(np.arange(len(val_loss)) * 10,
                        val_loss, label="Validation loss")
                plt.pause(0.001)

        torch.save(net.state_dict(),
                   MLP.SAVE_PATH + "/" + network_id + "/trained_network" + network_id + "final")

    @staticmethod
    def run_batch(i, x, y, train, net, optimizer, batch_size):
        input_x = torch.tensor(x[i * batch_size: (i + 1) * batch_size],
                               dtype=torch.float, device=device)
        target_y = torch.tensor(y[i * batch_size: (i + 1) * batch_size], dtype=torch.float, device=device,
                                requires_grad=False)

        optimizer.zero_grad()
        predict_y = net.forward(input_x)
        loss = MLP.loss_function(target_y, predict_y)

        if train:
            loss.backward()
            optimizer.step()

        return loss.item()

    @staticmethod
    def loss_function(target_y, predict_y):
        criterion = nn.MSELoss()
        mse = criterion(predict_y, target_y)

        return mse

    def load_model(self, model_id):
        self.load_state_dict(torch.load(os.path.join(
            self.SAVE_PATH, model_id+"/trained_network"+model_id)))
        self.eval()

    def predict_y(self, fep_agent: FepAgent):
        x = torch.Tensor([fep_agent.a[0, 0], fep_agent.a[0, 1],
                          fep_agent.a_dot[0, 0], fep_agent.a_dot[0, 1],
                          fep_agent.mu[0, 0], fep_agent.mu[0, 1],
                          fep_agent.s_p[0, 0], fep_agent.s_p[0, 1],
                          fep_agent.env.get_cartesian_distance(), ])
        return self.forward(x)
