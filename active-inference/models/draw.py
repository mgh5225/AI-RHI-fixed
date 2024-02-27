import torch
import torch.nn as nn
import torch.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from utils.utility import *
from utils.configs import draw_configs

from torch.utils.tensorboard import SummaryWriter


class DrawModel(nn.Module):
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "trained_draw")

    def __init__(self, T, A, B, z_size, N, dec_size, enc_size):
        super(DrawModel, self).__init__()
        self.T = T
        # self.batch_size = batch_size
        self.A = A
        self.B = B
        self.z_size = z_size
        self.N = N
        self.dec_size = dec_size
        self.enc_size = enc_size
        self.cs = [0] * T
        self.logsigmas, self.sigmas, self.mus = [0] * T, [0] * T, [0] * T

        self.encoder = nn.LSTMCell(2 * N * N + dec_size, enc_size)
        self.encoder_gru = nn.GRUCell(2 * N * N + dec_size, enc_size)
        self.mu_linear = nn.Linear(dec_size, z_size)
        self.sigma_linear = nn.Linear(dec_size, z_size)

        self.decoder = nn.LSTMCell(z_size, dec_size)
        self.decoder_gru = nn.GRUCell(z_size, dec_size)
        self.dec_linear = nn.Linear(dec_size, 5)
        self.dec_w_linear = nn.Linear(dec_size, N*N)

        self.sigmoid = nn.Sigmoid()

    def normalSample(self):
        return Variable(torch.randn(self.batch_size, self.z_size))

    # correct
    def compute_mu(self, g, rng, delta):
        rng_t, delta_t = align(rng, delta)
        tmp = (rng_t - self.N / 2 - 0.5) * delta_t
        tmp_t, g_t = align(tmp, g)
        mu = tmp_t + g_t
        return mu

    # correct
    def filterbank(self, gx, gy, sigma2, delta):
        rng = Variable(torch.arange(0, self.N).view(1, -1))
        mu_x = self.compute_mu(gx, rng, delta)
        mu_y = self.compute_mu(gy, rng, delta)

        a = Variable(torch.arange(0, self.A).view(1, 1, -1))
        b = Variable(torch.arange(0, self.B).view(1, 1, -1))

        mu_x = mu_x.view(-1, self.N, 1)
        mu_y = mu_y.view(-1, self.N, 1)
        sigma2 = sigma2.view(-1, 1, 1)

        Fx = self.filterbank_matrices(a, mu_x, sigma2)
        Fy = self.filterbank_matrices(b, mu_y, sigma2)

        return Fx, Fy

    def forward(self, x):
        self.batch_size = x.size()[0]
        h_dec_prev = Variable(torch.zeros(self.batch_size, self.dec_size))
        h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size))

        enc_state = Variable(torch.zeros(self.batch_size, self.enc_size))
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size))
        for t in xrange(self.T):
            c_prev = Variable(torch.zeros(
                self.batch_size, self.A * self.B)) if t == 0 else self.cs[t-1]
            x_hat = x - self.sigmoid(c_prev)     # 3
            r_t = self.read(x, x_hat, h_dec_prev)
            h_enc_prev, enc_state = self.encoder(
                torch.cat((r_t, h_dec_prev), 1), (h_enc_prev, enc_state))
            # h_enc = self.encoder_gru(torch.cat((r_t,h_dec_prev),1),h_enc_prev)
            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(
                h_enc_prev)
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            # h_dec = self.decoder_gru(z, h_dec_prev)
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec

    def loss(self, x):
        self.forward(x)
        criterion = nn.BCELoss()
        x_recons = self.sigmoid(self.cs[-1])
        Lx = criterion(x_recons, x) * self.A * self.B
        Lz = 0
        kl_terms = [0] * T
        for t in xrange(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]
            # Lz += (0.5 * (mu_2 + sigma_2 - 2 * logsigma))    # 11
            kl_terms[t] = 0.5 * \
                torch.sum(mu_2+sigma_2-2 * logsigma, 1) - self.T * 0.5
            Lz += kl_terms[t]
        # Lz -= self.T / 2
        Lz = torch.mean(Lz)
        loss = Lz + Lx    # 12
        return loss

    # correct
    def filterbank_matrices(self, a, mu_x, sigma2, epsilon=1e-9):
        t_a, t_mu_x = align(a, mu_x)
        temp = t_a - t_mu_x
        temp, t_sigma = align(temp, sigma2)
        temp = temp / (t_sigma * 2)
        F = torch.exp(-torch.pow(temp, 2))
        F = F / (F.sum(2, True).expand_as(F) + epsilon)
        return F

    # correct
    def attn_window(self, h_dec):
        params = self.dec_linear(h_dec)
        gx_, gy_, log_sigma_2, log_delta, log_gamma = params.split(1, 1)  # 21

        # gx_ = Variable(torch.ones(4,1))
        # gy_ = Variable(torch.ones(4, 1) * 2)
        # log_sigma_2 = Variable(torch.ones(4, 1) * 3)
        # log_delta = Variable(torch.ones(4, 1) * 4)
        # log_gamma = Variable(torch.ones(4, 1) * 5)

        gx = (self.A + 1) / 2 * (gx_ + 1)    # 22
        gy = (self.B + 1) / 2 * (gy_ + 1)    # 23
        delta = (max(self.A, self.B) - 1) / \
            (self.N - 1) * torch.exp(log_delta)  # 24
        sigma2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self.filterbank(gx, gy, sigma2, delta), gamma
    # correct

    def read(self, x, x_hat, h_dec_prev):
        (Fx, Fy), gamma = self.attn_window(h_dec_prev)

        def filter_img(img, Fx, Fy, gamma, A, B, N):
            Fxt = Fx.transpose(2, 1)
            img = img.view(-1, B, A)
            # img = img.transpose(2,1)
            # glimpse = matmul(Fy,matmul(img,Fxt))
            glimpse = Fy.bmm(img.bmm(Fxt))
            glimpse = glimpse.view(-1, N*N)
            return glimpse * gamma.view(-1, 1).expand_as(glimpse)
        x = filter_img(x, Fx, Fy, gamma, self.A, self.B, self.N)
        x_hat = filter_img(x_hat, Fx, Fy, gamma, self.A, self.B, self.N)
        return torch.cat((x, x_hat), 1)

    # correct
    def write(self, h_dec=0):
        w = self.dec_w_linear(h_dec)
        w = w.view(self.batch_size, self.N, self.N)
        # w = Variable(torch.ones(4,5,5) * 3)
        # self.batch_size = 4
        (Fx, Fy), gamma = self.attn_window(h_dec)
        Fyt = Fy.transpose(2, 1)
        # wr = matmul(Fyt,matmul(w,Fx))
        wr = Fyt.bmm(w.bmm(Fx))
        wr = wr.view(self.batch_size, self.A*self.B)
        return wr / gamma.view(-1, 1).expand_as(wr)

    def sampleQ(self, h_enc):
        e = self.normalSample()
        # mu_sigma = self.mu_sigma_linear(h_enc)
        # mu = mu_sigma[:, :self.z_size]
        # log_sigma = mu_sigma[:, self.z_size:]
        mu = self.mu_linear(h_enc)           # 1
        log_sigma = self.sigma_linear(h_enc)  # 2
        sigma = torch.exp(log_sigma)

        return mu + sigma * e, mu, log_sigma, sigma

    def generate(self, batch_size=64):
        self.batch_size = batch_size
        h_dec_prev = Variable(torch.zeros(
            self.batch_size, self.dec_size), volatile=True)
        dec_state = Variable(torch.zeros(
            self.batch_size, self.dec_size), volatile=True)

        for t in xrange(self.T):
            c_prev = Variable(torch.zeros(
                self.batch_size, self.A * self.B)) if t == 0 else self.cs[t - 1]
            z = self.normalSample()
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec
        imgs = []
        for img in self.cs:
            imgs.append(self.sigmoid(img).cpu().data.numpy())
        return imgs

    @staticmethod
    def train_net(net: 'DrawModel', config: draw_configs, dataset):
        torch.cuda.empty_cache()
        writer = SummaryWriter("runs/" + config.network_name)

        train_size = int(config.split_at * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(
            dataset,
            [train_size, test_size]
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            drop_last=config.drop_last
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            drop_last=config.drop_last
        )

        optimizer = optim.Adam(
            net.parameters(),
            lr=config.lr,
            betas=(config.b1, config.b2)
        )

        scheduler = StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma
        )

        epoch_loss = []
        val_loss = []

        for epoch in range(max_epochs):
            cur_batch_loss = []
            cur_val_loss = []

            train_iter = iter(train_dataloader)
            test_iter = iter(test_dataloader)

            for (x, _) in train_iter:
                loss = DrawModel.run_batch(
                    net,
                    x,
                    True,
                    optimizer,
                    config
                )
                cur_batch_loss = np.append(cur_batch_loss, [loss])

            scheduler.step()

            epoch_loss = np.append(epoch_loss, [np.mean(cur_batch_loss)])

            if epoch % 10 == 0 or epoch == (max_epochs - 1):

                for (x, _) in test_iter:
                    loss = DrawModel.run_batch(
                        net,
                        x,
                        False,
                        optimizer,
                        config
                    )
                    cur_val_loss = np.append(cur_val_loss, [loss])
                val_loss = np.append(val_loss, [np.mean(cur_val_loss)])

                print('------ Epoch ', epoch, '-------- LR:',
                      scheduler.get_last_lr())
                print('Epoch loss:', epoch_loss[-1])
                print('Val loss:', val_loss[-1])

                writer.add_scalar("Loss/train", epoch_loss[-1], epoch)
                writer.add_scalar("Loss/validation", val_loss[-1], epoch)

        torch.save(net.state_dict(),
                   DrawModel.SAVE_PATH + "/" + network_id + "/trained_network" + network_id + "final")

    @staticmethod
    def run_batch(net: 'DrawModel', input_x: torch.Tensor, is_train: bool, optimizer: optim.Optimizer, config: draw_configs):
        optimizer.zero_grad()

        bs = input_x.size()[0]
        data = Variable(input_x).view(bs, -1)

        loss = net.loss(data)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.clip)
            optimizer.step()

        return loss.item()

    def load_from_file(self, model_id):
        self.load_state_dict(torch.load(os.path.join(
            self.SAVE_PATH, model_id+"/trained_network"+model_id+"final")))
        self.eval()
        self.double()
