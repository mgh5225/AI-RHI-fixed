import numpy as np
import os

from utils import configs
from utils.fep_agent import FepAgent
from models.main import MLP
from models.vae import VAE_CNN
from unity.enums import *
from unity.environment import UnityContainer


editor_mode = 0
model_id = "vae"
mode_prefix = 'init_mu_centered_'
dir_name = 'inference'
log_id = "test"
log_path = os.path.join(os.path.dirname(__file__), "operation_logs/")


def rhi_task(condition: Condition, stimulation: Stimulation):
    n_iterations = 1500
    unity = UnityContainer(editor_mode)
    unity.initialise_environment()

    visual_decoder = VAE_CNN()
    visual_decoder.load_from_file(model_id)
    unity.set_condition(condition)
    unity.set_stimulation(stimulation)
    unity.set_visible_arm(VisibleArm.RubberArm)
    unity.reset()

    mode_name = f"{mode_prefix}{condition.name.lower()}_{stimulation.name.lower()}"

    # The proper range for ball is (0, 0.15)
    ball_ranges = [
        BallRange(0, 0.05),
        BallRange(0, 0.15),
        BallRange(0.1, 0.15),
    ]

    data_range = np.load(visual_decoder.SAVE_PATH + "/" +
                         model_id + "/data_range" + model_id + ".npy")

    agent = FepAgent(
        unity,
        visual_decoder,
        data_range,
        enable_action=False,
        init_mu=True
    )

    for ball_range in ball_ranges:
        unity.set_ball_range(ball_range)
        unity.reset()

        dir_name_new = dir_name + f"_{ball_range.b_min}_{ball_range.b_max}"
        agent.run_simulation(
            log_id, log_path, n_iterations, dir_name_new, mode_name)

    unity.close()


def full_rhi_task(condition: Condition, stimulation: Stimulation, with_mu: bool, plot: bool):
    n_iterations = 1500

    mlp_configs = configs.mlp_configs

    unity = UnityContainer(editor_mode)
    unity.initialise_environment()

    visual_decoder = VAE_CNN()
    visual_decoder.load_from_file(model_id)

    mlp = MLP(mlp_configs.input_size, mlp_configs.output_size,
              mlp_configs.hidden_layers)
    mlp.load_model(mlp_configs.name)

    unity.set_condition(condition)
    unity.set_stimulation(stimulation)
    unity.set_visible_arm(VisibleArm.RubberArm)
    unity.set_mode(Mode.Inference)
    unity.reset()

    data_range = np.load(visual_decoder.SAVE_PATH + "/" +
                         model_id + "/data_range" + model_id + ".npy")

    agent = FepAgent(unity, visual_decoder, data_range, enable_action=False)

    mode_name = f"{mode_prefix}{condition.name.lower()}_{stimulation.name.lower()}"

    agent.run_simulation(log_id, log_path, n_iterations, dir_name, mode_name)

    if plot:
        mlp.plot_y(n_iterations, agent)
    else:
        yh = mlp.predict_y(agent, with_mu)
        print("Predicted y: ", yh.item())

    unity.close()


# Example RHI task
rhi_task(Condition.Left, Stimulation.Synchronous)
rhi_task(Condition.Left, Stimulation.Asynchronous)

rhi_task(Condition.Center, Stimulation.Synchronous)
rhi_task(Condition.Center, Stimulation.Asynchronous)

rhi_task(Condition.Right, Stimulation.Synchronous)
rhi_task(Condition.Right, Stimulation.Asynchronous)
