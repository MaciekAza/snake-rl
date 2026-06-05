import random
import pickle
import os
from pathlib import Path

from project_paths import Q_TABLE_FILE
from rl.environment import ACTIONS


class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        return self.best_action(state)

    def best_action(self, state):
        values = self.get_values(state)
        best = ACTIONS[0]

        for action in ACTIONS:
            if values[action] > values[best]:
                best = action

        return best

    def learn(self, state, action, reward, next_state, game_over):
        values = self.get_values(state)
        old_value = values[action]

        if game_over:
            next_best_value = 0
        else:
            next_values = self.get_values(next_state)
            next_best_value = max(next_values.values())

        new_value = old_value + self.learning_rate * (
            reward + self.discount * next_best_value - old_value
        )

        values[action] = new_value

    def get_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = {}

            for action in ACTIONS:
                self.q_table[state][action] = 0

        return self.q_table[state]

    def lower_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay

        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def save(self, filename=Q_TABLE_FILE):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Tablica Q zapisana do {filename}")

    def load(self, filename=Q_TABLE_FILE):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Tablica Q wczytana z {filename}")
            return True
        return False
