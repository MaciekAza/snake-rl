import csv
from pathlib import Path

from agents.q_learning import QLearningAgent
from project_paths import Q_LEARNING_HISTORY_FILE, Q_TABLE_FILE
from rl.evaluation import evaluate_rl, print_evaluation
from rl.environment import SnakeRLEnvironment
from settings import BOARD_HEIGHT, BOARD_WIDTH, MAX_STEPS, Q_LEARNING_EPISODES, TEST_GAMES


RESET_Q_TABLE = False
FIELDNAMES = ["epizod", "wynik", "kroki", "epsilon", "rozmiar_tablicy_q"]


def load_last_training_state(history_path):
    if not history_path.exists():
        return 0, None

    last_row = None

    with history_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            last_row = row

    if last_row is None:
        return 0, None

    try:
        episode = int(last_row["epizod"])
    except (KeyError, ValueError):
        episode = 0

    try:
        epsilon = float(last_row["epsilon"])
    except (KeyError, ValueError):
        epsilon = None

    return episode, epsilon


def train():
    env = SnakeRLEnvironment(width=BOARD_WIDTH, height=BOARD_HEIGHT, max_steps=MAX_STEPS)
    agent = QLearningAgent()
    history_path = Path(Q_LEARNING_HISTORY_FILE)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    start_episode = 0

    if RESET_Q_TABLE:
        print("Start od pustej Q-table")
    else:
        loaded = agent.load(Q_TABLE_FILE)

        if loaded:
            start_episode, saved_epsilon = load_last_training_state(history_path)

            if saved_epsilon is not None:
                agent.epsilon = max(agent.epsilon_min, saved_epsilon)

            print(f"Kontynuacja treningu od epizodu {start_episode}")
        else:
            print("Nie znaleziono Q-table, start od pustej tablicy")

    file_mode = "a" if start_episode > 0 and not RESET_Q_TABLE else "w"

    with history_path.open(file_mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        if file_mode == "w":
            writer.writeheader()

        for episode in range(start_episode, Q_LEARNING_EPISODES):
            state = env.reset()

            while not env.game.game_over:
                action = agent.choose_action(state)
                next_state, reward, game_over = env.step(action)
                agent.learn(state, action, reward, next_state, game_over)
                state = next_state

            agent.lower_epsilon()

            writer.writerow(
                {
                    "epizod": episode + 1,
                    "wynik": env.game.score,
                    "kroki": env.game.steps,
                    "epsilon": f"{agent.epsilon:.6f}",
                    "rozmiar_tablicy_q": len(agent.q_table),
                }
            )

            if (episode + 1) % 500 == 0:
                print(
                    f"epizod {episode + 1}/{Q_LEARNING_EPISODES}, "
                    f"epsilon {agent.epsilon:.4f}, "
                    f"rozmiar tablicy Q: {len(agent.q_table)}"
                )

    agent.save(Q_TABLE_FILE)
    print(f"Historia treningu zapisana do {Q_LEARNING_HISTORY_FILE}")
    return agent


def test(agent):
    metrics = evaluate_rl(
        agent,
        games=TEST_GAMES,
        width=BOARD_WIDTH,
        height=BOARD_HEIGHT,
        max_steps=MAX_STEPS,
    )
    print_evaluation("Test Q-learning", metrics)
    return metrics


def main():
    agent = train()
    test(agent)


if __name__ == "__main__":
    main()
