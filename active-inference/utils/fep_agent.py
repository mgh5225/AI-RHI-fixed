from statistics import mean
from scipy.stats import norm
import torch
import numpy as np
import matplotlib.pyplot as plt

from .csv_logger import CSVLogger
from .functions import min_max_norm_dr, add_gaussian_noise

from models.vae import VAE_CNN
from unity.environment import UnityContainer
from unity.enums import *

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FepAgent:
    """
    Class representing the free-energy principle agent

    Adapted from the PixelAI algorithm, created by Cansu Sancaktar
    See https://github.com/cansu97/PixelAI for the latest PixelAI version
    """

    # Number of joints used
    N_JOINTS = 2

    # Latent variable size
    N_LATENT = 3

    # Image parameters
    IMG_WIDTH = 256
    IMG_HEIGHT = 256

    # Minimum number of visuo-tactile stimulation to get illusion

    def __init__(
        self,
        environment: UnityContainer,
        visual_decoder: VAE_CNN,
        data_range,
        enable_action: bool,
        attractor_image=None,
        min_iter_for_illusion=100,
        init_mu=False
    ):
        """
        Initialise the agent
        :param environment: environment the agent resides in
        :param visual_decoder: visual decoder model
        :param data_range: data range corresponding to the visual decoder
        :param attractor_image: optional visual attractor
        """
        self.env = environment
        self.visual_decoder = visual_decoder
        self.data_range = data_range
        self.action_enabled = enable_action
        self.init_mu = init_mu

        # Initialise belief vector
        self.mu = np.zeros((1, self.N_LATENT))
        self.mu_dot = np.zeros((1, self.N_LATENT))

        # Initialise action vector
        self.a = np.zeros((1, self.N_JOINTS))
        self.a_dot = np.zeros((1, self.N_JOINTS))

        # Initialise proprioception vector
        self.s_p = np.zeros((1, self.N_LATENT))
        # Initialise visual perception matrix
        self.s_v = np.zeros((1, 1, self.IMG_HEIGHT, self.IMG_WIDTH))

        # Initialise visual prediction matrix
        self.g_mu = np.zeros((1, 1, self.IMG_HEIGHT, self.IMG_WIDTH))

        # Initialise attractor image matrix
        self.attractor_im = np.zeros((1, 1, self.IMG_HEIGHT, self.IMG_WIDTH))
        if attractor_image is not None:
            self.attractor_im[0, 0] = attractor_image

        """FEP parameters"""
        self.a_clamp = 1

        # Time difference per step (Unity's default fixed update step is 0.02)
        self.dt = 0.02
        self.sigma_v_mu = 1*1e4
        self.sigma_v_a = 1*1e3

        self.sp_noise_variance = 0
        self.sigma_p = 1
        self.sigma_mu = 5 * 1e3

        self.attractor_active = 0
        self.beta = 1
        self.prior_gamma = 0.01
        self.gamma = self.prior_gamma
        """/FEP parameters"""

        """Parameter tracking lists"""
        self.mu_s_tracker = []
        self.mu_e_tracker = []
        self.mu_dot_s_tracker = []
        self.mu_dot_e_tracker = []
        self.a_s_tracker = []
        self.a_e_tracker = []
        self.a_dot_s_tracker = []
        self.a_dot_e_tracker = []
        self.vis_err_tracker = []
        self.attr_error_tracker = []
        self.gamma_tracker = []
        self.vt_tracker = []
        self.tt_tracker = []
        """/Parameter tracking lists"""

        """Visuo-tactile stimulation parameters"""
        self.last_tv = -60
        self.last_tt = -60
        self.new_tv = False
        self.new_tt = False
        self.r_decay = 1 / 80
        """/Visuo-tactile stimulation parameters"""

        """Additional paramters"""
        self.plot_interval = 50
        """/Additional paramters"""

        """Parameters for Illusion"""
        self.min_diff_mu = 5*1e-5
        self.illusion = torch.tensor([])
        self.min_iter_for_illusion = min_iter_for_illusion
        """/Parameters for Illusion"""

    def get_visual_forward(self, inpt):
        """
        Generate visual forward prediction based on (predicted) joint angles
        :param inpt: (predicted) joint angles
        :return: input and output tensors (input used for backward pass)
        """
        inpt = torch.tensor(min_max_norm_dr(inpt, self.data_range), device=device, dtype=torch.float,
                            requires_grad=True)
        output = self.visual_decoder.visual_prediction(inpt)
        return inpt, output

    def get_df_dmu_vis(self, inpt, output):
        """
        Get the visual part of the partial derivative of the free energy (F) with respect to belief (mu)
        by back-propagating the visual prediction error through the visual decoder
        :param inpt: input tensor from the get_visual_forward function
        :param output: output tensor from the get_visual_forward function
        :return: numpy array of shape (1, N_LATENT) containing the visual part of dF/dmu
        """
        neg_dF_dg = torch.tensor(
            (1/self.sigma_v_mu) * self.pred_error, dtype=torch.float, device=device)

        # Model the influence of the visual perception through visuo-tactile stimulation
        neg_dF_dg_rhi = self.gamma * neg_dF_dg

        # Set the gradient to zero before the backward pass to make sure there is no accumulation of gradients
        inpt.grad = torch.zeros(
            inpt.size(), device=device, dtype=torch.float, requires_grad=False)
        output.backward(neg_dF_dg_rhi, retain_graph=True)
        return inpt.grad.data.cpu().numpy()

    def get_df_dmu_attr(self, inpt, output):
        """
        Get the attractor part of the partial derivative of the free energy (F) with respect to belief (mu)
        by back-propagating the visual prediction error (with respect to the attractor) through the visual decoder.
        :param inpt: input tensor from the get_visual_forward function
        :param output: output tensor from the get_visual_forward function
        :return: numpy array of shape (1, N_LATENT) containing the attractor part of dF/dmu
        """
        attr_error = self.attractor_im - self.g_mu

        """tracking"""
        self.attr_error_tracker = np.array([np.square(attr_error).mean()])
        """/tracking"""

        # Set the gradient to zero before the backward pass to make sure there is no accumulation of gradients
        inpt.grad = torch.zeros(
            inpt.size(), device=device, dtype=torch.float, requires_grad=False)

        # To compute the full Jacobian for debugging purposes, uncomment the code below.
        # Note that it takes a significant time for this computation to complete:
        # np.save("./jac", compute_jacobian(inpt, output[0]).cpu().detach())

        output.backward(torch.tensor(self.beta*attr_error *
                        (1/self.sigma_mu), device=device), retain_graph=True)
        return inpt.grad.data.cpu().numpy()

    def get_df_da_visual(self, dF_dmu_vis):
        """
        Get the visual part of the partial derivative of the free energy (F) with respect to the action (a)
        by inverting the visual part of the partial derivative of the free energy (F) with respect to the belief (mu)
        :param dF_dmu_vis: partial derivative of the free energy (F) with respect to the belief (mu)
        :return: numpy array of shape (1, N_LATENT) containing the visual part of dF/da
        """
        return (-1) * dF_dmu_vis

    def get_posterior_gamma(self, last_touch_events):
        """
        Get the new value for gamma based on visuo-tactile stimulation events and the decay.
        :param last_touch_events: array containing the timepoints of the last touch events (visual and tactile)
        as reported by the environment
        :return: new value for gamma
        """
        if last_touch_events[0, 0] != self.last_tv:
            self.last_tv = last_touch_events[0, 0]
            self.new_tv = True
            self.vt_tracker.append(self.last_tv / self.dt)
        if last_touch_events[0, 1] != self.last_tt:
            self.last_tt = last_touch_events[0, 1]
            self.new_tt = True
            self.tt_tracker.append(self.last_tt/self.dt)

        if self.new_tv and self.new_tt:
            print("New Visuo-tactile stimulation event! VT:", self.last_tv, "TT", self.last_tt, "Diff (ms)",
                  round((self.last_tt - self.last_tv) * 1000))
            p_tv_tt_c1 = norm.pdf(self.last_tv - self.last_tt, 0, 0.2)
            p_tv_tt_c2 = 0.5  # uniform: 1/(b-a) = 1/(1--1) = 1/2
            new_gamma = p_tv_tt_c1 * self.gamma / \
                (p_tv_tt_c1 * self.gamma + p_tv_tt_c2 * (1 - self.gamma))
            self.new_tv = False
            self.new_tt = False
        else:
            new_gamma = (self.gamma * np.exp(-((self.env.get_current_env_time() -
                         self.last_tt) ** 2 / (1 / self.dt)) * self.r_decay))[0]

        return min(max(self.prior_gamma, new_gamma), 1)

    def get_observation(self):
        joint_observation = self.env.get_joint_observation()
        ball_observation = self.env.get_active_ball_distance()

        return np.append(joint_observation, [ball_observation], axis=1)

    def active_inference_step(self):
        """
        Perform one active inference update step
        """
        inpt, output = self.get_visual_forward(self.mu)
        self.g_mu = output.cpu().data.numpy()
        self.pred_error = self.s_v - self.g_mu

        # dF/dmu using visual information:
        dF_dmu_vis = self.get_df_dmu_vis(inpt, output)

        if self.attractor_active:
            # dF/dmu with attractor:
            self.mu_dot = dF_dmu_vis + self.get_df_dmu_attr(inpt, output)
        else:
            self.mu_dot = dF_dmu_vis

        # Proprioception
        self.mu_dot += (1 / self.sigma_p) * (self.s_p - self.mu)

        # Update mu:
        self.mu = self.mu + self.dt * self.mu_dot

        # Compute the action:
        self.a_dot = np.zeros((1, self.N_JOINTS))
        self.a_dot += (-(1 / self.sigma_p) *
                       (self.s_p - self.mu))[:, 0:self.N_JOINTS]

        # Update a:
        self.a = self.a + self.a_dot * self.dt
        self.a = np.clip(self.a, -self.a_clamp, self.a_clamp)

        """tracking"""
        self.vis_err_tracker.append(np.square(self.pred_error).mean())
        self.mu_s_tracker.append(self.mu[0, 0])
        self.mu_e_tracker.append(self.mu[0, 1])
        self.a_s_tracker.append(self.a[0, 0])
        self.a_e_tracker.append(self.a[0, 1])

        self.mu_dot_s_tracker.append(self.mu_dot[0, 0])
        self.mu_dot_e_tracker.append(self.mu_dot[0, 1])
        self.a_dot_s_tracker.append(self.a_dot[0, 0])
        self.a_dot_e_tracker.append(self.a_dot[0, 1])
        """/tracking"""

    def run_simulation(self, log_id: str, log_path: str, n_iterations: int, dir_name: str, mode_name: str):
        """
        Run a simulation by iteratively performing updating steps
        :param log_id: file identifier that will be used to write operation_logs to.
                       If not specified, no log will be created
        :param n_iterations: number of iteration to run the simulation for
        """
        if log_id is not None:
            csv_logger = CSVLogger(log_id, log_path)
            csv_logger.write_header(self)

        b_writer = SummaryWriter(f"runs/{dir_name}/belief/{mode_name}")
        p_writer = SummaryWriter(f"runs/{dir_name}/perception/{mode_name}")

        if self.init_mu:
            env_condition = self.env.env_condition

            self.env.set_condition(Condition.Center)
            self.env.reset()

            o_mu = self.get_mu_observation()
            self.mu = o_mu[np.newaxis, ...]

            self.env.set_condition(env_condition)
            self.env.reset()

        for i in range(n_iterations):
            self.s_p = add_gaussian_noise(
                self.get_observation(), 0, self.sp_noise_variance)
            self.s_v[0, 0] = np.squeeze(self.env.get_visual_observation())
            self.gamma = self.get_posterior_gamma(
                self.env.get_touch_observation())
            self.gamma_tracker.append(self.gamma)

            self.active_inference_step()

            if i > self.min_iter_for_illusion:

                diff_mu_s = abs(self.mu_s_tracker[-1] - self.mu_s_tracker[-2])
                diff_mu_e = abs(self.mu_e_tracker[-1] - self.mu_e_tracker[-2])

                diff_mean_mu = mean([diff_mu_s, diff_mu_e])

                if diff_mean_mu <= self.min_diff_mu:
                    self.illusion = torch.cat(
                        (self.illusion, torch.tensor([1])))
                else:
                    self.illusion = torch.cat(
                        (self.illusion, torch.tensor([0])))
            else:
                self.illusion = torch.cat((self.illusion, torch.tensor([0])))

            # Image
            b_writer.add_image("Eye", self.g_mu, i, dataformats='NCHW')
            p_writer.add_image(
                "Eye", self.s_v[0, 0], i, dataformats='HW')

            b_writer.add_scalar("Mu - shoulder", self.mu[0, 0], i)
            b_writer.add_scalar("Mu - elbow", self.mu[0, 1], i)
            b_writer.add_scalar("Mu - height", self.mu[0, 2], i)

            p_writer.add_scalar("Mu - shoulder", self.s_p[0, 0], i)
            p_writer.add_scalar("Mu - elbow", self.s_p[0, 1], i)
            p_writer.add_scalar("Mu - height", self.s_p[0, 2], i)

            b_writer.add_scalar("a - shoulder", self.a[0, 0], i)
            b_writer.add_scalar("a - elbow", self.a[0, 1], i)

            b_writer.add_scalar("Gamma", self.gamma, i)

            if i % 10 == 0:
                print(
                    "Iteration", i,
                    "action:", self.a,
                    "belief", self.mu,
                    "GT", self.s_p,
                    "Ev attr", self.attr_error_tracker,
                    "Is Illusion", self.illusion[-1].item())
            if log_id is not None:
                csv_logger.write_iteration(self, i)

            if self.action_enabled:
                self.env.act(self.a)
            else:
                self.env.act(np.zeros((1, self.N_JOINTS)))

        plt.close()

    def get_mu_observation(self):
        visual_observation = torch.from_numpy(
            self.env.get_visual_observation()).to(device)

        visual_observation = visual_observation.permute((2, 0, 1)).double()

        output = self.visual_decoder.get_z(
            visual_observation.unsqueeze(0))

        o_mu = (output[0]).data.cpu().numpy()

        return o_mu
