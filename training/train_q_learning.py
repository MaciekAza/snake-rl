import csv
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from agents.q_learning import QLearningAgent
from project_paths import Q_LEARNING_HISTORY_FILE, Q_TABLE_FILE
from rl.environment import SnakeRLEnvironment


WIDTH = 10
HEIGHT = 10
MAX_STEPS = 1000
EPISODES = 30000
TEST_GAMES = 100
TRAINING_HISTORY_FILE = Q_LEARNING_HISTORY_FILE
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
    env = SnakeRLEnvironment(width=WIDTH, height=HEIGHT, max_steps=MAX_STEPS)
    agent = QLearningAgent()
    history_path = Path(TRAINING_HISTORY_FILE)
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

        for episode in range(start_episode, EPISODES):
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
                print(f"epizod {episode + 1}/{EPISODES}, epsilon {agent.epsilon:.4f}, rozmiar tablicy Q: {len(agent.q_table)}")

    agent.save(Q_TABLE_FILE)
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
    print("Test Q-learning")
    print("=" * 50)
    print(f"  średni wynik: {avg_score:.2f}")
    print(f"  najlepszy wynik: {max(scores)}")
    print(f"  najgorszy wynik: {min(scores)}")
    print(f"  średnia liczba kroków: {avg_steps:.2f}")
    print(f"  liczba testów: {TEST_GAMES}")
    print("=" * 50)


def main():
    agent = train()
    test(agent)


if __name__ == "__main__":
    main()
