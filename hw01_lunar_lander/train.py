from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4

SEED = 42


class DQN:
    def __init__(self, state_dim, action_dim, buffer_size=10000):
        self.steps = 0  # Do not change
        self.memory = deque(maxlen=buffer_size)

        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        self.target_model = copy.deepcopy(self.model)

        self.optimizer = Adam(self.model.parameters(), LEARNING_RATE)

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.memory.append(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        batch = random.sample(self.memory, BATCH_SIZE)
        return list(zip(*batch))

    def train_step(self, batch):
        # Use batch to update DQN's network.
        state, action, next_state, reward, done = self.sample_batch()

        state = torch.tensor(np.array(state, dtype=np.float32))
        action = torch.tensor(np.array(action, dtype=np.int64))
        next_state = torch.tensor(np.array(next_state, dtype=np.float32))
        reward = torch.tensor(np.array(reward, dtype=np.float32))
        done = torch.tensor(np.array(done, dtype=np.float32))

        # print(action.shape)
        # print(self.model(state).shape)

        current_q_value = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        # print(current_q_value.shape)

        target_next_q_value = self.target_model(next_state).max(dim=1)[0].detach()
        # print(target_next_q_value.shape)
        # target_next_q_value = target_next_q_value.unsqueeze(1)
        # print(target_next_q_value.shape)

        q_expected = reward + GAMMA * target_next_q_value * (1 - done)

        # print(current_q_value.shape) [128,1]
        # print(q_expected.shape) #[128]
        # q_expected = q_expected.unsqueeze(1)
        # print(q_expected.shape) #[128]
        # print(current_q_value.shape)

        # loss = F.mse_loss(current_q_value, q_expected)
        crit = nn.MSELoss(reduce=False)
        loss = crit(current_q_value, q_expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        # print(state)
        # print(state.shape)
        state = np.array(state, dtype=np.float32)
        state = torch.tensor(state)
        action = self.model(state).argmax().numpy()
        return action

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            # print("updating target")
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.0

        for _ in range(1000):
            while not done:
                state, reward, done, *_ = env.step(agent.act(state))
                total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":

    # SEED

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = make("LunarLander-v2")
    env.seed(SEED)

    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()

    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, *_ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

    for i in range(TRANSITIONS):
        # Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, *_ = env.step(action)
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(
                f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}"
            )
            if np.mean(rewards) > 200:
                dqn.save()
