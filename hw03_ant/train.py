import pybullet_envs
from gym import make
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import random
import copy

# для лучшей стабильности обучения
# - поменял MSE на Smooth L1 loss
# - добавил L2 регуляризацию на все сетки

# пробовал
# - добавить batch norm
# но видимо не получилось правильно расставить eval() и train()

GAMMA = 0.99
TAU = 0.002
EPS = 0.2
WEIGHT_DECAY = 1e-5
CRITIC_LR = 5e-4
ACTOR_LR = 2e-4
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 128
ENV_NAME = "AntBulletEnv-v0"
TRANSITIONS = 1_000_000


def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ELU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.model(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ELU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ELU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class TD3:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)

        self.actor_optim = Adam(
            self.actor.parameters(),
            lr=ACTOR_LR,
            weight_decay=WEIGHT_DECAY,
        )
        self.critic_1_optim = Adam(
            self.critic_1.parameters(),
            lr=CRITIC_LR,
            weight_decay=WEIGHT_DECAY,
        )
        self.critic_2_optim = Adam(
            self.critic_2.parameters(),
            lr=CRITIC_LR,
            weight_decay=WEIGHT_DECAY,
        )

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.replay_buffer = deque(maxlen=200000)

    def update(self, transition):
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > BATCH_SIZE * 16:

            # Sample batch
            transitions = [
                self.replay_buffer[random.randint(0, len(self.replay_buffer) - 1)]
                for _ in range(BATCH_SIZE)
            ]
            state, action, next_state, reward, done = zip(*transitions)
            state = torch.tensor(np.array(state), device=DEVICE, dtype=torch.float)
            action = torch.tensor(np.array(action), device=DEVICE, dtype=torch.float)
            next_state = torch.tensor(
                np.array(next_state), device=DEVICE, dtype=torch.float
            )
            reward = torch.tensor(np.array(reward), device=DEVICE, dtype=torch.float)
            done = torch.tensor(np.array(done), device=DEVICE, dtype=torch.float)

            # Update critic
            with torch.no_grad():  # убираем градиент, чтобы не текли градиенты по ответам
                noise = torch.randn_like(action).to(DEVICE)  # шум для выбора действия
                next_actor_action = torch.clamp(
                    self.target_actor(next_state) + EPS * noise, -1, 1
                )  # добавляем шума для эксплорейшена политики
                # предсказываем обоими критиками и берём меньшее значение
                target_q1 = self.target_critic_1(next_state, next_actor_action)
                target_q2 = self.target_critic_2(next_state, next_actor_action)
                target_min_q = reward + (1 - done) * GAMMA * torch.min(
                    target_q1, target_q2
                )

            # 1 критик
            self.critic_1_optim.zero_grad()
            curr_q1 = self.critic_1(state, action)
            loss_cr_1 = F.smooth_l1_loss(curr_q1, target_min_q)
            loss_cr_1.backward()
            self.critic_1_optim.step()

            # 2 критик
            self.critic_2_optim.zero_grad()
            curr_q2 = self.critic_2(state, action)
            loss_cr_2 = F.smooth_l1_loss(curr_q2, target_min_q)
            loss_cr_2.backward()
            self.critic_2_optim.step()

            # актор
            self.actor_optim.zero_grad()
            actor_action = self.actor(state)
            loss_actor = -self.critic_1(state, actor_action).mean()
            loss_actor.backward()
            self.actor_optim.step()

            # Update actor

            soft_update(self.target_critic_1, self.critic_1)
            soft_update(self.target_critic_2, self.critic_2)
            soft_update(self.target_actor, self.actor)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=DEVICE)
            return self.actor(state).cpu().numpy()[0]

    def save(self):
        torch.save(
            self.actor.state_dict(),
            "/content/drive/MyDrive/Colab Notebooks/made_2022_practice_rl/made_2022_practice_rl_hw_03/agent.pkl",
        )


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.0

        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make(ENV_NAME)
    test_env = make(ENV_NAME)
    td3 = TD3(
        state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]
    )
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0

    max_mean_rew = -100
    curr_mean_rew = None

    for i in range(TRANSITIONS):
        steps = 0

        # Epsilon-greedy policy
        action = td3.act(state)
        action = np.clip(action + EPS * np.random.randn(*action.shape), -1, +1)

        next_state, reward, done, _ = env.step(action)
        td3.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()

        if (i + 1) % (TRANSITIONS // 300) == 0:
            rewards = evaluate_policy(test_env, td3, 15)
            curr_mean_rew = np.mean(rewards)
            curr_std_rew = np.std(rewards)
            print(
                f"Step: {i+1}, Reward mean: {curr_mean_rew}, Reward std: {curr_std_rew}"
            )
            # если среднее за 15 проигрываний стало выше, то сохраняем новую модель актора (агента)
            if curr_mean_rew > max_mean_rew:
                td3.save()
                max_mean_rew = curr_mean_rew
