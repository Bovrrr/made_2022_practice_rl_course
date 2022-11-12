import random
import numpy as np
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1909
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(SEED)
print(f"Random seed set as {SEED}")


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float()
            action, *_ = self.model.act(state)
            return action.cpu().numpy()  # TODO

    def reset(self):
        pass
