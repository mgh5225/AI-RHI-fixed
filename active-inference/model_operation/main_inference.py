import numpy as np
import os

from utils import configs
from utils.csv_logger import CSVLogger
from utils.fep_agent import FepAgent
from models.main import MLP
from models.vae import VAE_CNN
from unity.enums import *
from unity.environment import UnityContainer


editor_mode = 0
model_id = "vae"

log_id = "test"
log_path = os.path.join(os.path.dirname(__file__), "operation_logs/")


def reaching_tasks(model, position_import_model_id=None):
    ids = [model + "reachclose", model + 'reachfar']

    positions = np.zeros((2, 40, 2))
    if position_import_model_id is None:
        positions[0] = np.random.uniform(low=1, high=8, size=(
            40, 2))*(-2 * np.random.randint(low=0, high=2, size=(40, 2)) + 1)
        positions[1] = np.random.uniform(low=8, high=18, size=(
            40, 2))*(-2 * np.random.randint(low=0, high=2, size=(40, 2)) + 1)
    else:
        positions[0] = CSVLogger.import_log(log_id=position_import_model_id + "reach" + "close" +
                                            "0n", n=40, length=5000,
                                            columns=["rubb_Shoulder", "rubb_Elbow"])[:, 0, :]

        positions[1] = CSVLogger.import_log(log_id=position_import_model_id + "reach" + "far" +
                                            "0n", n=40,  length=5000,
                                            columns=["rubb_Shoulder", "rubb_Elbow"])[:, 0, :]

    n_iterations = 5000

    for i in range(0, 40):
        for c in range(positions.shape[0]):
            unity = UnityContainer(editor_mode)
            unity.initialise_environment()

            visual_decoder = VAE_CNN()
            visual_decoder.load_from_file(model_id)

            unity.set_visible_arm(VisibleArm.RealArm)
            unity.reset()

            unity.set_rubber_arm_rotation(np.array([[-17, -10]]))

            # Set real arm to rubber arm position to get attr image
            unity.set_rotation(unity.get_rubber_joint_observation())
            attr_image = np.squeeze(unity.get_visual_observation())
            unity.set_rotation(np.zeros((1, 2)))

            data_range = np.load(
                visual_decoder.SAVE_PATH + "/" + model_id + "/data_range" + model_id + ".npy")
            central = FepAgent(unity, visual_decoder, data_range,
                               enable_action=True, attractor_image=attr_image)
            log_id = ids[c] + str(central.sp_noise_variance) + "n" + str(i)

            central.run_simulation(log_id, log_path, n_iterations)
            unity.close()


def rhi_task(condition, stimulation):
    n_iterations = 1500
    unity = UnityContainer(editor_mode)
    unity.initialise_environment()

    visual_decoder = VAE_CNN()
    visual_decoder.load_from_file(model_id)
    unity.set_condition(condition)
    unity.set_stimulation(stimulation)
    unity.set_visible_arm(VisibleArm.RubberArm)
    unity.reset()

    data_range = np.load(visual_decoder.SAVE_PATH + "/" +
                         model_id + "/data_range" + model_id + ".npy")
    central = FepAgent(unity, visual_decoder, data_range, enable_action=False)

    central.run_simulation(log_id, log_path, n_iterations)
    unity.close()


def full_rhi_task(condition, stimulation, with_mu):
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
    unity.reset()

    data_range = np.load(visual_decoder.SAVE_PATH + "/" +
                         model_id + "/data_range" + model_id + ".npy")

    agent = FepAgent(unity, visual_decoder, data_range, enable_action=False)
    agent.run_simulation(log_id, log_path, n_iterations)

    yh = mlp.predict_y(agent, with_mu)
    print("Predicted y: ", yh.item())

    unity.close()


# Example RHI task
# rhi_task(Condition.Left, Stimulation.Asynchronous)
full_rhi_task(Condition.Right, Stimulation.Synchronous, True)
# full_rhi_task(Condition.Left, Stimulation.Asynchronous, False)
