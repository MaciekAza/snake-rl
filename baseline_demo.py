from agents.baseline import FoodAgent, RandomAgent, SafeRandomAgent
from snake_game.game import SnakeGame


def play_one_game(agent, width=10, height=10, max_steps=500):
    game = SnakeGame(width=width, height=height)

    while not game.game_over and game.steps < max_steps:
        direction = agent.choose_direction(game)
        game.step(direction)

    return game.score, game.steps


def test_agent(agent, games=100):
    scores = []
    steps = []

    for _ in range(games):
        score, step_count = play_one_game(agent)
        scores.append(score)
        steps.append(step_count)

    average_score = sum(scores) / len(scores)
    best_score = max(scores)
    average_steps = sum(steps) / len(steps)

    print(agent.name)
    print(f"  średni wynik: {average_score:.2f}")
    print(f"  najlepszy wynik: {best_score}")
    print(f"  średnia liczba kroków: {average_steps:.2f}")
    print()


def main():
    agents = [
        RandomAgent(),
        SafeRandomAgent(),
        FoodAgent(),
    ]

    for agent in agents:
        test_agent(agent)


if __name__ == "__main__":
    main()
