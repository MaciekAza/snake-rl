import csv
import importlib.util

from agents.baseline import FoodAgent, RandomAgent
from agents.dqn import DQNAgent
from agents.q_learning import QLearningAgent
from project_paths import (
    DQN_MODEL_FILE,
    EXPERIMENT_RESULTS_FILE,
    NEAT_CONFIG_FILE,
    NEAT_HISTORY_FILE,
    NEAT_MODEL_FILE,
    Q_TABLE_FILE,
)
from rl.evaluation import evaluate_baseline, evaluate_rl
from rl.environment import ACTIONS, SnakeRLEnvironment
from settings import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    DQN_EPISODES,
    MAX_STEPS,
    NEAT_GENERATIONS,
    Q_LEARNING_EPISODES,
    TEST_GAMES,
)


RESULT_FIELDS = [
    "agent",
    "średni_wynik",
    "najlepszy_wynik",
    "średnie_kroki",
    "epizody_treningu",
    "liczba_testów",
    "plansza",
    "limit_kroków",
]


def make_result(name, metrics, training_episodes):
    return {
        "agent": name,
        "średni_wynik": f"{metrics['average_score']:.2f}",
        "najlepszy_wynik": metrics["best_score"],
        "średnie_kroki": f"{metrics['average_steps']:.2f}",
        "epizody_treningu": training_episodes,
        "liczba_testów": metrics["games"],
        "plansza": f"{BOARD_WIDTH}x{BOARD_HEIGHT}",
        "limit_kroków": MAX_STEPS,
    }


def evaluate_agent(name, agent, training_episodes, rl_agent):
    evaluator = evaluate_rl if rl_agent else evaluate_baseline
    metrics = evaluator(
        agent,
        games=TEST_GAMES,
        width=BOARD_WIDTH,
        height=BOARD_HEIGHT,
        max_steps=MAX_STEPS,
    )
    return make_result(name, metrics, training_episodes)


def load_q_learning_agent():
    agent = QLearningAgent()
    agent.load(Q_TABLE_FILE)
    return agent


def load_dqn_agent():
    env = SnakeRLEnvironment(width=BOARD_WIDTH, height=BOARD_HEIGHT, max_steps=MAX_STEPS)
    state_size = len(env.reset())
    agent = DQNAgent(state_size=state_size, action_size=len(ACTIONS))
    agent.load(DQN_MODEL_FILE)
    return agent


def load_neat_agent():
    if importlib.util.find_spec("neat") is None:
        print("Nie znaleziono biblioteki neat-python. Uruchom: pip install -r requirements.txt")
        return None

    if not NEAT_MODEL_FILE.exists():
        print("Nie znaleziono modelu NEAT. Najpierw uruchom: python -m training.train_neat")
        return None

    if not NEAT_CONFIG_FILE.exists():
        print("Nie znaleziono konfiguracji NEAT.")
        return None

    from agents.neat_agent import NEATAgent

    return NEATAgent.load(NEAT_MODEL_FILE, NEAT_CONFIG_FILE)


def get_neat_training_generations():
    if not NEAT_HISTORY_FILE.exists():
        return NEAT_GENERATIONS

    text = NEAT_HISTORY_FILE.read_text(encoding="utf-8").replace("\x00", "")
    generations = []

    for row in csv.DictReader(text.splitlines()):
        try:
            generations.append(int(row["pokolenie"]))
        except (KeyError, ValueError):
            continue

    return max(generations, default=NEAT_GENERATIONS)


def save_results(results):
    path = EXPERIMENT_RESULTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wyniki zapisane do {EXPERIMENT_RESULTS_FILE}")


def main():
    results = [
        evaluate_agent("losowy", RandomAgent(), training_episodes=0, rl_agent=False),
        evaluate_agent("heurystyka_jedzenia", FoodAgent(), training_episodes=0, rl_agent=False),
        evaluate_agent("Q-learning", load_q_learning_agent(), Q_LEARNING_EPISODES, rl_agent=True),
        evaluate_agent("DQN", load_dqn_agent(), DQN_EPISODES, rl_agent=True),
    ]

    neat_agent = load_neat_agent()

    if neat_agent is not None:
        results.append(
            evaluate_agent(
                "NEAT",
                neat_agent,
                get_neat_training_generations(),
                rl_agent=True,
            )
        )

    save_results(results)

    for result in results:
        print(
            f"{result['agent']}: "
            f"średni wynik {result['średni_wynik']}, "
            f"najlepszy {result['najlepszy_wynik']}, "
            f"średnie kroki {result['średnie_kroki']}, "
            f"epizody treningu {result['epizody_treningu']}"
        )


if __name__ == "__main__":
    main()
