import csv
from pathlib import Path

from agents.q_learning import QLearningAgent
from rl.environment import SnakeRLEnvironment


EPISODES = 10000
TEST_GAMES = 100
Q_TABLE_FILE = "q_table.pkl"
TRAINING_HISTORY_FILE = "results/q_learning_training.csv"
RESET_Q_TABLE = True


def train():
    env = SnakeRLEnvironment(width=10, height=10)
    agent = QLearningAgent()
    
    if RESET_Q_TABLE:
        print("Start od pustej Q-table")
    else:
        agent.load(Q_TABLE_FILE)

    history_path = Path(TRAINING_HISTORY_FILE)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epizod", "wynik", "kroki", "epsilon", "rozmiar_tablicy_q"],
        )
        writer.writeheader()

        for episode in range(EPISODES):
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
    env = SnakeRLEnvironment(width=10, height=10)
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
