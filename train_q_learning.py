from agents.q_learning import QLearningAgent
from rl.environment import SnakeRLEnvironment


EPISODES = 10000
TEST_GAMES = 100
Q_TABLE_FILE = "q_table.pkl"


def train():
    env = SnakeRLEnvironment(width=10, height=10)
    agent = QLearningAgent()
    
    # Spróbuj wczytać wcześniej trenowaną Q-table
    agent.load(Q_TABLE_FILE)

    for episode in range(EPISODES):
        state = env.reset()

        while not env.game.game_over:
            action = agent.choose_action(state)
            next_state, reward, game_over = env.step(action)
            agent.learn(state, action, reward, next_state, game_over)
            state = next_state

        agent.lower_epsilon()

        if (episode + 1) % 500 == 0:
            print(f"epizod {episode + 1}/{EPISODES}, epsilon {agent.epsilon:.4f}, Q-table size: {len(agent.q_table)}")

    # Zapisz wytrenowaną Q-table
    agent.save(Q_TABLE_FILE)
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
    print("Q-learning test")
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
