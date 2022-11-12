import random
import numpy as np
from torch import nn
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
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


class MyModel(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        lin_dim_list=[
            int(2**9),
            int(2**8),
        ],
    ):
        super().__init__()
        layers = []
        self.lin_dim_list = [state_dim] + lin_dim_list
        for i in range(len(self.lin_dim_list) - 1):
            layers.append(nn.Linear(self.lin_dim_list[i], self.lin_dim_list[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(self.lin_dim_list[i + 1]))

        layers.append(nn.Linear(lin_dim_list[-1], action_dim))
        self.fcnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.fcnn(x)

    @staticmethod
    def weights_init(layer):
        classname = layer.__class__.__name__
        if classname.find("Linear") != -1:
            nn.init.xavier_uniform_(layer.weight)


class Agent:
    def __init__(self):
        self.model = MyModel(8, 4)
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        self.model.eval()

    def act(self, state):
        return self.model(torch.tensor(state).unsqueeze(0)).max(1)[1].view(1, 1).item()


agent = Agent()
