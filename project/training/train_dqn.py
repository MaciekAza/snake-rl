import csv
import random
from pathlib import Path

import torch

from project.agents.dqn import DQNAgent
from project.environment import ACTIONS, SnakeRLEnvironment
from project.evaluation import evaluate_rl, print_evaluation
from project.paths import DQN_HISTORY_FILE, DQN_MODEL_FILE
from project.settings import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DQN_EPISODES,
    MAX_STEPS,
    RANDOM_SEED,
    TEST_GAMES,
)


LEARN_EVERY_STEPS = 4
TARGET_UPDATE_EVERY_EPISODES = 20


def train():
    torch.set_num_threads(1)
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    env = SnakeRLEnvironment(
        width=BOARD_WIDTH,
        height=BOARD_HEIGHT,
        max_steps=MAX_STEPS,
        seed=RANDOM_SEED,
    )
    state_size = len(env.reset())
    agent = DQNAgent(state_size=state_size, action_size=len(ACTIONS))

    history_path = Path(DQN_HISTORY_FILE)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    temp_history_path = history_path.with_suffix(".tmp.csv")

    with temp_history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epizod", "wynik", "kroki", "epsilon", "rozmiar_pamięci", "średnia_strata"],
        )
        writer.writeheader()

        for episode in range(DQN_EPISODES):
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
                    f"epizod {episode + 1}/{DQN_EPISODES}, "
                    f"epsilon {agent.epsilon:.4f}, "
                    f"pamięć {len(agent.memory)}"
                )

    temp_history_path.replace(history_path)
    agent.save(DQN_MODEL_FILE)
    print(f"Historia treningu zapisana do {DQN_HISTORY_FILE}")
    return agent


def test(agent):
    metrics = evaluate_rl(
        agent,
        games=TEST_GAMES,
        width=BOARD_WIDTH,
        height=BOARD_HEIGHT,
    )
    print_evaluation("Test DQN", metrics)
    return metrics


def main():
    agent = train()
    test(agent)


if __name__ == "__main__":
    main()
