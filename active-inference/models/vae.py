import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import os
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
import numpy as np
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VAE_CNN(nn.Module):
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "trained_vae_cnn")

    def __init__(self):
        super(VAE_CNN, self).__init__()

        # Encoder
        self.e_conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.e_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.e_conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.e_conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.e_conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.e_conv5 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.e_fc1 = nn.Linear(8 * 8 * 128, 4096)
        self.e_fc2 = nn.Linear(4096, 1024)
        self.e_fc3 = nn.Linear(1024, 512)

        # Variational latent variable layers
        self.fc_mu = nn.Linear(512, 3)
        self.fc_logvar = nn.Linear(512, 3)

        # Decoder
        self.d_fc1 = nn.Linear(3, 1024)
        self.d_fc2 = nn.Linear(1024, 8 * 8 * 128)

        self.d_upconv1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.d_conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.d_upconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.d_conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.d_upconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.d_conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.d_upconv4 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.d_conv4 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.d_upconv5 = nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.to(device)
        self.double()

    def encode(self, x):
        """
        Run the encoder (first part of the forward pass)
        :param x: input image
        :return: mu and logvar vector
        """
        x = self.relu(self.e_pool(self.e_conv1(x)))
        x = self.relu(self.e_pool(self.e_conv2(x)))
        x = self.relu(self.e_pool(self.e_conv3(x)))
        x = self.relu(self.e_pool(self.e_conv4(x)))
        x = self.relu(self.e_pool(self.e_conv5(x)))

        # Reshaping the output of the fully conv layer so that it is compatible with the fc layers
        x = x.view(-1, 8 * 8 * 128)

        x = self.relu(self.e_fc1(x))
        x = self.relu(self.e_fc2(x))
        x = self.relu(self.e_fc3(x))

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Return latent parameters
        return mu, logvar

    @staticmethod
    def sample_z(mu, logvar):
        """
        Randomly sample z based on mu and logvar vectors
        :param mu: mu vector
        :param logvar: logvar vector
        :return: z
        """
        eps = torch.randn_like(logvar)
        return eps * torch.exp(0.5 * logvar) + mu

    def visual_prediction(self, z):
        """
        Get visual prediction for a set of joint angles
        :param z: latent variable
        :return: visual prediction
        """
        return self.decode(z)

    def get_z(self, x):
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        return z

    def decode(self, z):
        """
        Run the decoder (second part of the forward pass)
        :param z: latent variable vector z
        :return: output image
        """
        # Two fully connected layers of neurons:
        x = self.relu(self.d_fc1(z.double()))
        x = self.relu(self.d_fc2(x))

        # Reshaping the output of the fully connected layer so that it is compatible with the conv layers
        x = x.view(-1, 128, 8, 8)

        # Upsampling using the deconvolutional layers:
        x = self.relu(self.d_upconv1(x))
        x = self.relu(self.d_conv1(x))

        x = self.relu(self.d_upconv2(x))
        x = self.relu(self.d_conv2(x))

        x = self.relu(self.d_upconv3(x))
        x = self.relu(self.d_conv3(x))

        x = self.relu(self.d_upconv4(x))
        x = self.relu(self.d_conv4(x))

        x = self.sigmoid(self.d_upconv5(x))
        return x

    def forward(self, x):
        """
        Perform forward pass through the network
        :param x: input
        :return: network output, mu and logvar vectors
        """
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar

    @staticmethod
    def train_net(net: nn.Module, dataset, network_id, max_epochs=600, batch_size=125):
        """
        Train the neural network
        :param net: the network object
        :param X: Input samples
        :param Y: Output samples
        :param network_id: network id for saving
        :param max_epochs: max number of epochs to train for
        :param batch_size: size of the mini-batches
        """
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
                loss, _ = VAE_CNN.run_batch(x, y, True, net, optimizer)
                cur_batch_loss = np.append(cur_batch_loss, [loss])

            scheduler.step()

            epoch_loss = np.append(epoch_loss, [np.mean(cur_batch_loss)])

            if epoch % 10 == 0 or epoch == (max_epochs - 1):

                for (x, y) in test_iter:
                    loss, _ = VAE_CNN.run_batch(x, y, False, net, optimizer)
                    cur_val_loss = np.append(cur_val_loss, [loss])
                val_loss = np.append(val_loss, [np.mean(cur_val_loss)])

                print('------ Epoch ', epoch, '-------- LR:',
                      scheduler.get_last_lr())
                print('Epoch loss:', epoch_loss[-1])
                print('Val loss:', val_loss[-1])

                writer.add_scalar("Loss/train", epoch_loss[-1], epoch)
                writer.add_scalar("Loss/validation", val_loss[-1], epoch)

        torch.save(net.state_dict(),
                   VAE_CNN.SAVE_PATH + "/" + network_id + "/trained_network" + network_id + "final")

    @staticmethod
    def run_batch(input_x, target_y, train, net, optimizer):
        optimizer.zero_grad()
        predict_y, mu, logvar = net.forward(target_y)
        loss = VAE_CNN.loss_function(target_y, predict_y, input_x, mu, logvar)

        if train:
            loss.backward()
            optimizer.step()

        return loss.item(), predict_y

    @staticmethod
    def loss_function(target_y, predict_y, q, mu, logvar):
        """
        Loss function of the VAE, based on the MSE of the images and regularisation term
        based on the Kullback-Leibler divergence
        :param target_y: target image
        :param predict_y: predicted image
        :param q: the ground truth joint angles
        :param mu: the latent mean joint angle vector
        :param logvar: the latent log variance joint angle vector
        :return: loss
        """
        criterion = nn.MSELoss()
        mse = criterion(predict_y, target_y)
        target_var = torch.zeros(
            logvar.shape, dtype=torch.float, device=device)
        target_var[:, :] = 0.001
        grand_truth_angles = q.squeeze()
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - torch.log(target_var) -
                                          (logvar.exp() + (grand_truth_angles - mu) ** 2)/target_var, dim=1), dim=0)

        return mse + kld

    def load_from_file(self, model_id):
        """
        Load network from file
        :param model_id: save id to load from
        """
        self.load_state_dict(torch.load(os.path.join(
            self.SAVE_PATH, model_id+"/trained_network"+model_id+"final")))
        self.eval()
        self.double()
