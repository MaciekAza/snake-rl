from agents.baseline import FoodAgent, RandomAgent, SafeRandomAgent
from rl.evaluation import evaluate_baseline, print_evaluation
from settings import BOARD_HEIGHT, BOARD_WIDTH, MAX_STEPS, TEST_GAMES


def main():
    for agent in (RandomAgent(), SafeRandomAgent(), FoodAgent()):
        metrics = evaluate_baseline(
            agent,
            games=TEST_GAMES,
            width=BOARD_WIDTH,
            height=BOARD_HEIGHT,
            max_steps=MAX_STEPS,
        )
        print_evaluation(agent.name, metrics)


if __name__ == "__main__":
    main()
