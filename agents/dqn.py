import random
from collections import deque

import torch
from torch import nn

from rl.environment import ACTIONS


class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.layers(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)

        self.discount = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNetwork(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.update_target_network()

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        with torch.no_grad():
            state_tensor = self._state_tensor([state])
            q_values = self.model(state_tensor)
            action_index = torch.argmax(q_values, dim=1).item()

        return ACTIONS[action_index]

    def remember(self, state, action, reward, next_state, game_over):
        action_index = ACTIONS.index(action)
        self.memory.append((state, action_index, reward, next_state, game_over))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, game_overs = zip(*batch)

        states_tensor = self._state_tensor(states)
        next_states_tensor = self._state_tensor(next_states)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        game_overs_tensor = torch.tensor(game_overs, dtype=torch.float32, device=self.device)

        current_q = self.model(states_tensor).gather(1, actions_tensor).squeeze(1)

        with torch.no_grad():
            next_q = self.target_model(next_states_tensor).max(dim=1).values
            target_q = rewards_tensor + self.discount * next_q * (1 - game_overs_tensor)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def lower_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filename="dqn_model.pth"):
        torch.save(
            {
                "model": self.model.state_dict(),
                "target_model": self.target_model.state_dict(),
                "epsilon": self.epsilon,
            },
            filename,
        )
        print(f"DQN model zapisany do {filename}")

    def load(self, filename="dqn_model.pth"):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.target_model.load_state_dict(checkpoint["target_model"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        print(f"DQN model wczytany z {filename}")

    def _state_tensor(self, states):
        return torch.tensor(states, dtype=torch.float32, device=self.device)
