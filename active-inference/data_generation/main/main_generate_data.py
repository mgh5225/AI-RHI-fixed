import torch

from .data_generation import DataGeneration
from unity.enums import *
from unity.environment import UnityContainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device =", device)


editor_mode = 0
model_id = "vae"

data_gen = DataGeneration(model_id)

# Initialise Unity environment
unity = UnityContainer(editor_mode)
unity.initialise_environment()

# Left
data_id = "left_sync0"

unity.set_condition(Condition.Left)
unity.set_stimulation(Stimulation.Synchronous)
unity.set_visible_arm(VisibleArm.RubberArm)
unity.reset()

# Generate and save the data
data_gen.generate_data(unity, data_id, 800)

# Center
data_id = "center_sync0"

unity.set_condition(Condition.Center)
unity.set_stimulation(Stimulation.Synchronous)
unity.set_visible_arm(VisibleArm.RubberArm)
unity.reset()

# Generate and save the data
data_gen.generate_data(unity, data_id, 550)

# Right
data_id = "right_sync0"

unity.set_condition(Condition.Right)
unity.set_stimulation(Stimulation.Synchronous)
unity.set_visible_arm(VisibleArm.RubberArm)
unity.reset()

# Generate and save the data
data_gen.generate_data(unity, data_id, 700)

# Close the Unity environment
unity.close()
