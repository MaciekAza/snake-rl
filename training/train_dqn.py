import csv
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import torch

from agents.dqn import DQNAgent
from project_paths import DQN_HISTORY_FILE, DQN_MODEL_FILE
from rl.environment import ACTIONS, SnakeRLEnvironment


WIDTH = 10
HEIGHT = 10
MAX_STEPS = 1000
EPISODES = 1500
TEST_GAMES = 100
LEARN_EVERY_STEPS = 4
TARGET_UPDATE_EVERY_EPISODES = 20
TRAINING_HISTORY_FILE = DQN_HISTORY_FILE


def train():
    torch.set_num_threads(1)

    env = SnakeRLEnvironment(width=WIDTH, height=HEIGHT, max_steps=MAX_STEPS)
    state_size = len(env.reset())
    agent = DQNAgent(state_size=state_size, action_size=len(ACTIONS))

    history_path = Path(TRAINING_HISTORY_FILE)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    temp_history_path = history_path.with_suffix(".tmp.csv")

    with temp_history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epizod", "wynik", "kroki", "epsilon", "rozmiar_pamięci", "średnia_strata"],
        )
        writer.writeheader()

        for episode in range(EPISODES):
            state = env.reset()
            losses = []
            step_counter = 0

            while not env.game.game_over:
                action = agent.choose_action(state)
                next_state, reward, game_over = env.step(action)
                agent.remember(state, action, reward, next_state, game_over)
                step_counter += 1
                loss = None

                if step_counter % LEARN_EVERY_STEPS == 0:
                    loss = agent.learn()

                if loss is not None:
                    losses.append(loss)

                state = next_state

            agent.lower_epsilon()

            if (episode + 1) % TARGET_UPDATE_EVERY_EPISODES == 0:
                agent.update_target_network()

            avg_loss = sum(losses) / len(losses) if losses else 0
            writer.writerow(
                {
                    "epizod": episode + 1,
                    "wynik": env.game.score,
                    "kroki": env.game.steps,
                    "epsilon": f"{agent.epsilon:.6f}",
                    "rozmiar_pamięci": len(agent.memory),
                    "średnia_strata": f"{avg_loss:.6f}",
                }
            )

            if (episode + 1) % 100 == 0:
                file.flush()
                print(
                    f"epizod {episode + 1}/{EPISODES}, "
                    f"epsilon {agent.epsilon:.4f}, "
                    f"pamięć {len(agent.memory)}"
                )

    temp_history_path.replace(history_path)
    agent.save(DQN_MODEL_FILE)
    print(f"Historia treningu zapisana do {TRAINING_HISTORY_FILE}")
    return agent


def test(agent):
    env = SnakeRLEnvironment(width=WIDTH, height=HEIGHT, max_steps=MAX_STEPS)
    old_epsilon = agent.epsilon
    agent.epsilon = 0

    scores = []
    steps = []

    for _ in range(TEST_GAMES):
        state = env.reset()

        while not env.game.game_over:
            action = agent.choose_action(state)
            state, reward, game_over = env.step(action)

        scores.append(env.game.score)
        steps.append(env.game.steps)

    agent.epsilon = old_epsilon

    avg_score = sum(scores) / len(scores)
    avg_steps = sum(steps) / len(steps)

    print()
    print("=" * 50)
    print("Test DQN")
    print("=" * 50)
    print(f"  średni wynik: {avg_score:.2f}")
    print(f"  najlepszy wynik: {max(scores)}")
    print(f"  najgorszy wynik: {min(scores)}")
    print(f"  średnia liczba kroków: {avg_steps:.2f}")
    print(f"  liczba testów: {TEST_GAMES}")
    print("=" * 50)

    return {
        "avg_score": avg_score,
        "best_score": max(scores),
        "avg_steps": avg_steps,
        "test_games": TEST_GAMES,
    }


def main():
    agent = train()
    test(agent)


if __name__ == "__main__":
    main()
