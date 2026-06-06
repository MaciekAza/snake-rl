from statistics import fmean

from rl.environment import SnakeRLEnvironment
from snake_game.game import SnakeGame


def summarize(scores, steps):
    return {
        "average_score": fmean(scores),
        "best_score": max(scores),
        "worst_score": min(scores),
        "average_steps": fmean(steps),
        "games": len(scores),
    }


def evaluate_baseline(agent, games, width, height, max_steps):
    scores = []
    steps = []

    for _ in range(games):
        game = SnakeGame(width=width, height=height)

        while not game.game_over and game.steps < max_steps:
            game.step(agent.choose_direction(game))

        scores.append(game.score)
        steps.append(game.steps)

    return summarize(scores, steps)


def choose_rl_action(agent, env, state):
    if hasattr(agent, "choose_action_from_env"):
        return agent.choose_action_from_env(env)

    return agent.choose_action(state)


def evaluate_rl(agent, games, width, height, max_steps):
    env = SnakeRLEnvironment(width=width, height=height, max_steps=max_steps)
    saved_epsilon = getattr(agent, "epsilon", None)

    if saved_epsilon is not None:
        agent.epsilon = 0

    scores = []
    steps = []

    try:
        for _ in range(games):
            state = env.reset()

            while not env.game.game_over:
                action = choose_rl_action(agent, env, state)
                state, _, _ = env.step(action)

            scores.append(env.game.score)
            steps.append(env.game.steps)
    finally:
        if saved_epsilon is not None:
            agent.epsilon = saved_epsilon

    return summarize(scores, steps)


def print_evaluation(title, metrics):
    print()
    print("=" * 50)
    print(title)
    print("=" * 50)
    print(f"  średni wynik: {metrics['average_score']:.2f}")
    print(f"  najlepszy wynik: {metrics['best_score']}")
    print(f"  najgorszy wynik: {metrics['worst_score']}")
    print(f"  średnia liczba kroków: {metrics['average_steps']:.2f}")
    print(f"  liczba testów: {metrics['games']}")
    print("=" * 50)
