from gym import make
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from collections import deque
import random

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = int(1e5) * 12
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
BUFFER_SIZE = int(2**7) * int(1e3)
LEARNING_RATE = 5e-4

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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


class ExpReplay(deque):
    def sample(self, size):
        batch = random.sample(self, size)
        return list(zip(*batch))


class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0  # Do not change
        self.model = MyModel(state_dim, action_dim).to(DEVICE)  # Torch model

        self.target_model = MyModel(state_dim, action_dim).to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())

        self.buffer = ExpReplay(maxlen=BUFFER_SIZE)

        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criteria = nn.MSELoss()
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(1e4),
            gamma=8e-1,
        )

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.buffer.append(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        # взяли батч из буффера
        batch = self.buffer.sample(BATCH_SIZE)
        # разложили его по компонентам
        state, action, next_state, reward, done = batch
        # оборачиваем в пайторч тензоры
        state = torch.tensor(np.array(state, dtype=np.float32))
        action = torch.tensor(np.array(action, dtype=np.int64))
        next_state = torch.tensor(np.array(next_state, dtype=np.float32))
        reward = torch.tensor(np.array(reward, dtype=np.float32))
        done = torch.tensor(np.array(done, dtype=np.int32))
        return state, action, next_state, reward, done

    def train_step(self, batch):
        # Use batch to update DQN's network.
        state, action, next_state, reward, done = batch

        state = torch.tensor(np.array(state, dtype=np.float32))
        action = torch.tensor(np.array(action, dtype=np.int64))
        next_state = torch.tensor(np.array(next_state, dtype=np.float32))
        reward = torch.tensor(np.array(reward, dtype=np.float32))
        done = torch.tensor(np.array(done, dtype=np.float32))

        current_q_value = (
            self.model(state.to(DEVICE))
            .gather(1, action.unsqueeze(1).to(DEVICE))
            .squeeze(1)
        ).to(DEVICE)
        target_next_q_value = (
            self.target_model(next_state.to(DEVICE)).max(dim=1)[0].detach()
        ).to(DEVICE)
        reward = reward.to(DEVICE)
        done = done.to(DEVICE)
        q_expected = reward + GAMMA * target_next_q_value * (1 - done)

        # loss = F.mse_loss(current_q_value, q_expected)
        # loss = F.smooth_l1_loss(current_q_value, q_expected)
        self.optimizer.zero_grad()
        loss = self.criteria(current_q_value, q_expected)
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        self.model.eval()
        network = self.target_model if target else self.model
        state = np.array(state)
        state = torch.tensor(state).view(1, -1).to(DEVICE)
        action_reward_collection = network(state).squeeze(0).detach().cpu().numpy()
        return np.argmax(action_reward_collection)

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model.state_dict(), "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    # ставим модель в состояние инференса
    agent.model.eval()
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.0

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)

    # возвращаем модель в состояние обучения
    agent.model.train()
    return returns


if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    eps_dec_coef = 2
    eps_min = 1e-2

    state = env.reset()

    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

    for i in range(TRANSITIONS):
        if 0 == i % 25000:
            eps = max(eps_min, eps / eps_dec_coef)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(
                f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}"
            )
            if np.mean(rewards) > 200:
                dqn.save()
