import torch

from .data_generation import DataGeneration
from unity.environment import UnityContainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device =", device)

editor_mode = 0
data_id = "s2nr"

data_gen = DataGeneration()

# Initialise Unity environment
unity = UnityContainer(editor_mode)
unity.initialise_environment()

# Generate and save the data
data_gen.generate_data(unity, data_id)

# Close the Unity environment
unity.close()
