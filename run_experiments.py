import csv
from pathlib import Path

from agents.baseline import FoodAgent, RandomAgent
from agents.dqn import DQNAgent
from agents.q_learning import QLearningAgent
from rl.environment import ACTIONS, SnakeRLEnvironment
from snake_game.game import SnakeGame
from train_dqn import DQN_MODEL_FILE, EPISODES as DQN_EPISODES
from train_q_learning import EPISODES as Q_LEARNING_EPISODES, Q_TABLE_FILE


WIDTH = 10
HEIGHT = 10
MAX_STEPS = 500
TEST_GAMES = 100
RESULTS_FILE = "results/experiment_results.csv"


def evaluate_baseline(agent, training_episodes):
    scores = []
    steps = []

    for _ in range(TEST_GAMES):
        game = SnakeGame(width=WIDTH, height=HEIGHT)

        while not game.game_over and game.steps < MAX_STEPS:
            direction = agent.choose_direction(game)
            game.step(direction)

        scores.append(game.score)
        steps.append(game.steps)

    return make_result(agent.name, scores, steps, training_episodes)


def evaluate_rl_agent(name, agent, training_episodes):
    env = SnakeRLEnvironment(width=WIDTH, height=HEIGHT, max_steps=MAX_STEPS)
    old_epsilon = getattr(agent, "epsilon", None)

    if old_epsilon is not None:
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

    if old_epsilon is not None:
        agent.epsilon = old_epsilon

    return make_result(name, scores, steps, training_episodes)


def make_result(name, scores, steps, training_episodes):
    return {
        "agent": name,
        "średni_wynik": f"{sum(scores) / len(scores):.2f}",
        "najlepszy_wynik": max(scores),
        "średnie_kroki": f"{sum(steps) / len(steps):.2f}",
        "epizody_treningu": training_episodes,
        "liczba_testów": TEST_GAMES,
        "plansza": f"{WIDTH}x{HEIGHT}",
        "limit_kroków": MAX_STEPS,
    }


def load_q_learning_agent():
    agent = QLearningAgent()
    agent.load(Q_TABLE_FILE)
    return agent


def load_dqn_agent():
    env = SnakeRLEnvironment(width=WIDTH, height=HEIGHT, max_steps=MAX_STEPS)
    state_size = len(env.reset())
    agent = DQNAgent(state_size=state_size, action_size=len(ACTIONS))
    agent.load(DQN_MODEL_FILE)
    return agent


def save_results(results):
    path = Path(RESULTS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "agent",
                "średni_wynik",
                "najlepszy_wynik",
                "średnie_kroki",
                "epizody_treningu",
                "liczba_testów",
                "plansza",
                "limit_kroków",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Wyniki zapisane do {RESULTS_FILE}")


def main():
    results = [
        evaluate_baseline(RandomAgent(), training_episodes=0),
        evaluate_baseline(FoodAgent(), training_episodes=0),
        evaluate_rl_agent("Q-learning", load_q_learning_agent(), Q_LEARNING_EPISODES),
        evaluate_rl_agent("DQN", load_dqn_agent(), DQN_EPISODES),
    ]

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
