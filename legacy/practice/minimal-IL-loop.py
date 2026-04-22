import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dm_control import suite

# Load Environment
env = suite.load('cartpole', 'swingup')

obs_spec = env.observation_spec()
act_spec = env.action_spec()

def flatten_obs(obs_dict):
    """Flatten dict of arrays into 1D numpy array."""
    return np.concatenate([v.ravel() for v in obs_dict.values()])

obs_dim = sum(np.prod(v.shape) for v in obs_spec.values())
act_dim = act_spec.shape[0]

# Collect dataset (replace this with expert policy later)
def collect_data(env, num_episodes=20):
    data = []
    for _ in range(num_episodes):
        ts = env.reset()
        while not ts.last():
            # Random expert (for demo); plug in real policy here
            action = np.random.uniform(act_spec.minimum,
                                       act_spec.maximum,
                                       size=act_spec.shape)
            next_ts = env.step(action)
            data.append((flatten_obs(ts.observation), action))
            ts = next_ts
    obs, acts = zip(*data)
    return np.array(obs), np.array(acts)

obs_data, act_data = collect_data(env)

# Define policy network
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.net(x)

policy = PolicyNet(obs_dim, act_dim)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Train behavior cloning
X = torch.tensor(obs_data, dtype=torch.float32)
Y = torch.tensor(act_data, dtype=torch.float32)

for epoch in range(50):
    pred = policy(X)
    loss = loss_fn(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.4f}")

# Rollout learned policy
ts = env.reset()
total_reward = 0
while not ts.last():
    obs = torch.tensor(flatten_obs(ts.observation), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = policy(obs).numpy().squeeze()
    ts = env.step(action)
    total_reward += ts.reward or 0
print("Total reward with learned policy:", total_reward)
