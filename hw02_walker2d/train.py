import pybullet_envs
import os

# Don't forget to install PyBullet!
from gym import make
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.optim import Adam
import random

# SEED FIXING, DEVICE DETERMINING
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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


ENV_NAME = "Walker2DBulletEnv-v0"

LAMBDA = 0.95
GAMMA = 0.99

ACTOR_LR = 2e-4
CRITIC_LR = 1e-4

CLIP = 0.2
ENTROPY_COEF = 1.5e-2
BATCHES_PER_UPDATE = 64
BATCH_SIZE = int(2**9)

MIN_TRANSITIONS_PER_UPDATE = 2048
MIN_EPISODES_PER_UPDATE = 4

ITERATIONS = 1000 * 2

CLIP_VALUE = 1
EPS = 1e-8


def compute_lambda_returns_and_gae(trajectory):
    lambda_returns = []
    gae = []
    last_lr = 0.0
    last_v = 0.0
    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)

    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [
        (s, a, p, v, adv)
        for (s, a, _, p, _), v, adv in zip(
            trajectory, reversed(lambda_returns), reversed(gae)
        )
    ]


# просто модель для Actor
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, input):
        return self.model(input)

    @staticmethod
    def weights_init(layer):
        classname = layer.__class__.__name__
        if classname.find("Linear") != -1:
            nn.init.xavier_uniform_(layer.weight)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Advice: use same log_sigma for all states to improve stability
        # You can do this by defining log_sigma as nn.Parameter(torch.zeros(...))
        self.model = Model(state_dim, action_dim).to(DEVICE)
        self.sigma = nn.Parameter(torch.zeros(action_dim)).to(DEVICE)

    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions
        mu = self.model(state.to(DEVICE))
        sigma = torch.exp(self.sigma)
        distrib = Normal(mu, sigma)  # batch_size x action_size
        return torch.exp(distrib.log_prob(action).sum(-1)), distrib

    def act(self, state):
        # Returns an action (with tanh), not-transformed action (without tanh) and distribution of non-transformed actions
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        mu = self.model(state.to(DEVICE))
        sigma = torch.exp(self.sigma)
        distrib = Normal(mu, sigma)
        action = distrib.sample()
        transform_action = torch.tanh(action)
        return transform_action, action, distrib


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = Model(state_dim, 1).to(DEVICE)

    def get_value(self, state):
        return self.model(state.to(DEVICE))


class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optim = Adam(self.actor.parameters(), ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), CRITIC_LR)

    def update(self, trajectories):
        transitions = [
            t for traj in trajectories for t in traj
        ]  # Turn a list of trajectories into list of transitions
        state, action, old_prob, target_value, advantage = zip(*transitions)
        state = np.array(state)
        action = np.array(action)
        old_prob = np.array(old_prob)
        target_value = np.array(target_value)
        advantage = np.array(advantage)
        # Нормализовать advantage при обучении actor’а
        advnatage = (advantage - advantage.mean()) / (advantage.std() + EPS)

        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(
                0, len(transitions), BATCH_SIZE
            )  # Choose random batch
            s = torch.tensor(np.array(state[idx])).float().to(DEVICE)
            a = torch.tensor(np.array(action[idx])).float().to(DEVICE)
            op = (
                torch.tensor(np.array(old_prob[idx])).float().to(DEVICE)
            )  # Probability of the action in state s.t. old policy
            v = (
                torch.tensor(np.array(target_value[idx])).float().to(DEVICE)
            )  # Estimated by lambda-returns
            adv = (
                torch.tensor(np.array(advantage[idx])).float().to(DEVICE)
            )  # Estimated by generalized advantage estimation

            # TODO: Update actor here
            new_prob, distrib = self.actor.compute_proba(s, a)
            ratio = torch.exp(torch.log(new_prob + EPS) - torch.log(op + EPS))
            y_ = ratio * adv
            y = torch.clip(ratio, 1 - CLIP, 1 + CLIP) * adv
            # заменяем мат. ожидание средним
            loss_actor = -torch.min(y_, y).mean()
            loss_actor_explo = loss_actor - ENTROPY_COEF * distrib.entropy().mean()

            self.actor_optim.zero_grad()
            loss_actor_explo.backward()
            nn.utils.clip_grad_norm_(
                self.actor.model.parameters(), CLIP_VALUE, norm_type=2
            )
            self.actor_optim.step()

            # TODO: Update critic here
            v_cr = self.critic.get_value(s).flatten()
            loss_critic = F.smooth_l1_loss(v_cr, v)

            self.critic_optim.zero_grad()
            loss_critic.backward()
            nn.utils.clip_grad_norm_(
                self.critic.model.parameters(), CLIP_VALUE, norm_type=2
            )
            self.critic_optim.step()

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            action, pure_action, distr = self.actor.act(state)
            prob = torch.exp(distr.log_prob(pure_action).sum(-1))
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], prob.cpu().item()

    def save(self):
        torch.save(self.actor, "agent.pkl")


def evaluate_policy(env, agent, episodes=50):  # заменил 5 на 50 для более точной оценки
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.0

        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns


def sample_episode(env, agent):
    s = env.reset()
    d = False
    trajectory = []
    while not d:
        a, pa, p = agent.act(s)
        v = agent.get_value(s)
        ns, r, d, _ = env.step(a)
        trajectory.append((s, pa, r, p, v))
        s = ns
    return compute_lambda_returns_and_gae(trajectory)


if __name__ == "__main__":
    env = make(ENV_NAME)
    ppo = PPO(
        state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]
    )
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0

    max_reward = -1e9

    for i in range(ITERATIONS):
        trajectories = []
        steps_ctn = 0

        while (
            len(trajectories) < MIN_EPISODES_PER_UPDATE
            or steps_ctn < MIN_TRANSITIONS_PER_UPDATE
        ):
            traj = sample_episode(env, ppo)
            steps_ctn += len(traj)
            trajectories.append(traj)
        episodes_sampled += len(trajectories)
        steps_sampled += steps_ctn

        ppo.update(trajectories)

        if (i + 1) % 50 == 0:
            rewards = evaluate_policy(env, ppo, 50)
            print(
                f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}, Episodes: {episodes_sampled}, Steps: {steps_sampled}"
            )
            if max_reward < np.mean(rewards):
                max_reward = np.mean(rewards)
                ppo.save()
