from utils.fep_agent import FepAgent
import os
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "trained_mlp")

    def __init__(self, input_size, output_size, hidden_layers: list = []):
        super(MLP, self).__init__()

        hidden = []
        active_fn = nn.Sigmoid()

        if output_size > 1:
            active_fn = nn.Softmax()

        for i in range(1, len(hidden_layers)):
            hidden.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            hidden.append(nn.BatchNorm1d(hidden_layers[i]))
            hidden.append(nn.LeakyReLU())

        if len(hidden_layers) == 0:
            self.layers = nn.Sequential(
                nn.Linear(input_size, output_size),
                active_fn
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_layers[0]),
                nn.BatchNorm1d(hidden_layers[0]),
                nn.LeakyReLU(),
                *hidden,
                nn.Linear(hidden_layers[-1], output_size),
                active_fn
            )

        self.to(device)
        self.double()

    def forward(self, x: torch.Tensor):
        x = x.to(device)
        y = self.layers(x)

        return y

    @staticmethod
    def train_model(net: nn.Module, dataset, network_id, max_epochs=600, batch_size=125):
        torch.cuda.empty_cache()
        writer = SummaryWriter("runs/" + network_id)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size])

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        optimizer = optim.Adam(net.parameters(), lr=0.001)  # 0.001
        scheduler = StepLR(optimizer, step_size=20, gamma=0.95)  # 0.95

        epoch_loss = []
        val_loss = []

        for epoch in range(max_epochs):
            cur_batch_loss = []
            cur_val_loss = []

            train_iter = iter(train_dataloader)
            test_iter = iter(test_dataloader)

            for (x, y) in train_iter:
                loss, _ = MLP.run_batch(x, y, True, net, optimizer)
                cur_batch_loss = np.append(cur_batch_loss, [loss])

            scheduler.step()

            epoch_loss = np.append(epoch_loss, [np.mean(cur_batch_loss)])

            if epoch % 10 == 0 or epoch == (max_epochs - 1):

                for (x, y) in test_iter:
                    loss, _ = MLP.run_batch(x, y, False, net, optimizer)
                    cur_val_loss = np.append(cur_val_loss, [loss])
                val_loss = np.append(val_loss, [np.mean(cur_val_loss)])

                print('------ Epoch ', epoch, '-------- LR:',
                      scheduler.get_last_lr())
                print('Epoch loss:', epoch_loss[-1])
                print('Val loss:', val_loss[-1])

                writer.add_scalar("Loss/train", epoch_loss[-1], epoch)
                writer.add_scalar("Loss/validation", val_loss[-1], epoch)

        torch.save(net.state_dict(),
                   MLP.SAVE_PATH + "/" + network_id + "/trained_network" + network_id)

        writer.flush()

        y_true = []
        y_pred = []

        test_iter = iter(test_dataloader)

        for (x, y) in test_iter:
            _, yh = MLP.run_batch(x, y, False, net, optimizer)

            y_true.extend(y.data.cpu().numpy())
            y_pred.extend(torch.round(yh).data.cpu().numpy())

        cf_matrix = confusion_matrix(y_true, y_pred)
        print(cf_matrix)

    @staticmethod
    def run_batch(input_x, target_y, train, net, optimizer):
        optimizer.zero_grad()
        predict_y = net.forward(input_x)
        loss = MLP.loss_function(target_y, predict_y)

        if train:
            loss.backward()
            optimizer.step()

        return loss.item(), predict_y

    @staticmethod
    def loss_function(target_y, predict_y):
        criterion = nn.BCELoss()
        bce = criterion(predict_y, target_y)

        return bce

    def load_model(self, model_id):
        self.load_state_dict(torch.load(os.path.join(
            self.SAVE_PATH, model_id+"/trained_network"+model_id)))
        self.eval()
        self.double()

    def predict_y(self, fep_agent: FepAgent, with_mu=False):
        if with_mu:
            o_mu = fep_agent.get_mu_observation()

            x = torch.Tensor([
                fep_agent.a[0, 0], fep_agent.a[0, 1],
                fep_agent.a_dot[0, 0], fep_agent.a_dot[0, 1],
                fep_agent.mu[0, 0], fep_agent.mu[0, 1],
                o_mu[0], o_mu[1]
            ]).double().unsqueeze(0)
            return self.forward(x)
        else:
            x = torch.Tensor([
                fep_agent.a[0, 0], fep_agent.a[0, 1],
                fep_agent.a_dot[0, 0], fep_agent.a_dot[0, 1],
                fep_agent.mu[0, 0], fep_agent.mu[0, 1]
            ]).double().unsqueeze(0)
            return self.forward(x)

    def plot_y(self, steps, fep_agent: FepAgent):
        writer = SummaryWriter("runs/yh")

        o_mu = fep_agent.get_mu_observation()

        for i in range(steps):
            x = torch.Tensor([
                fep_agent.a_s_tracker[i], fep_agent.a_e_tracker[i],
                fep_agent.a_dot_s_tracker[i], fep_agent.a_dot_e_tracker[i],
                fep_agent.mu_s_tracker[i], fep_agent.mu_e_tracker[i],
                o_mu[0], o_mu[1]
            ]).double().unsqueeze(0)

            yh = self.forward(x)
            writer.add_scalar("Predicted Y", yh, i)

        writer.flush()
