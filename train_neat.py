import csv
import pickle
from pathlib import Path

try:
    import neat
except ModuleNotFoundError:
    print("Brakuje biblioteki neat-python. Uruchom: pip install -r requirements.txt")
    raise SystemExit(1)

from agents.neat_agent import NEATAgent
from agents.neat_agent import build_neat_inputs
from project_paths import NEAT_CONFIG_FILE, NEAT_HISTORY_FILE, NEAT_MODEL_FILE
from rl.environment import ACTIONS, SnakeRLEnvironment


WIDTH = 10
HEIGHT = 10
MAX_STEPS = 1000
GENERATIONS = 35
EVALUATION_SEEDS = (101, 202, 303, 404, 505)
GAMES_PER_GENOME = len(EVALUATION_SEEDS)
MAX_STEPS_WITHOUT_FOOD = 80
TEST_GAMES = 100
CONFIG_FILE = NEAT_CONFIG_FILE
TRAINING_HISTORY_FILE = NEAT_HISTORY_FILE


def make_config():
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(CONFIG_FILE),
    )


def choose_action(network, env):
    outputs = network.activate(build_neat_inputs(env))
    safe_indexes = [index for index, action in enumerate(ACTIONS) if not env.is_danger(action)]

    if len(safe_indexes) == 0:
        safe_indexes = list(range(len(ACTIONS)))

    best_index = max(safe_indexes, key=lambda index: outputs[index])
    return ACTIONS[best_index]


def play_game(network, seed=None):
    env = SnakeRLEnvironment(width=WIDTH, height=HEIGHT, max_steps=MAX_STEPS, seed=seed)
    state = env.reset()
    fitness = 0
    steps_without_food = 0
    visited_states = {}

    while not env.game.game_over:
        old_score = env.game.score
        old_distance = env._calculate_food_distance()
        action = choose_action(network, env)
        state, reward, game_over = env.step(action)
        new_distance = env._calculate_food_distance()

        if env.game.score > old_score:
            steps_without_food = 0
            fitness += 1200 + env.game.score * 300
        else:
            steps_without_food += 1

            if new_distance < old_distance:
                fitness += 8
            elif new_distance > old_distance:
                fitness -= 6
            else:
                fitness -= 1

        head = env.game.snake[0]
        state_key = (head, env.game.food, env.game.direction, env.game.score)
        visited_states[state_key] = visited_states.get(state_key, 0) + 1

        if visited_states[state_key] > 2:
            fitness -= visited_states[state_key] * 4

        if steps_without_food > MAX_STEPS_WITHOUT_FOOD:
            fitness -= 400
            env.game.game_over = True
            env.game.reason = "stagnacja"

    fitness += env.game.score * 2500 + env.game.score * env.game.score * 900
    fitness -= steps_without_food * 1.2

    if env.game.score == 0:
        fitness -= 600

    if env.game.reason in ("wall", "body"):
        fitness -= 250
    elif env.game.reason in ("limit", "stagnacja"):
        fitness -= 300

    return env.game.score, env.game.steps, fitness


def evaluate_genome(genome, config):
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    scores = []
    steps = []
    fitness_values = []

    for seed in EVALUATION_SEEDS:
        score, step_count, fitness_value = play_game(network, seed)
        scores.append(score)
        steps.append(step_count)
        fitness_values.append(fitness_value)

    avg_score = sum(scores) / len(scores)
    avg_steps = sum(steps) / len(steps)
    fitness = sum(fitness_values) / len(fitness_values)

    genome.fitness = fitness
    genome.avg_score = avg_score
    genome.best_score = max(scores)
    genome.avg_steps = avg_steps

    return fitness


def train():
    config = make_config()
    population = neat.Population(config)
    population.add_reporter(neat.StatisticsReporter())

    history_path = Path(TRAINING_HISTORY_FILE)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    temp_history_path = history_path.with_suffix(".tmp.csv")
    generation = {"number": 0}

    with temp_history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "pokolenie",
                "najlepsza_sprawność",
                "średnia_sprawność",
                "średni_wynik_najlepszego",
                "średni_wynik_populacji",
                "najlepszy_wynik_najlepszego",
                "średnie_kroki_najlepszego",
                "liczba_genomów",
            ],
        )
        writer.writeheader()

        def eval_genomes(genomes, neat_config):
            generation["number"] += 1

            for genome_id, genome in genomes:
                evaluate_genome(genome, neat_config)

            scored_genomes = [genome for genome_id, genome in genomes]
            best = max(scored_genomes, key=lambda genome: genome.fitness)
            avg_fitness = sum(genome.fitness for genome in scored_genomes) / len(scored_genomes)
            avg_score = sum(genome.avg_score for genome in scored_genomes) / len(scored_genomes)

            writer.writerow(
                {
                    "pokolenie": generation["number"],
                    "najlepsza_sprawność": f"{best.fitness:.4f}",
                    "średnia_sprawność": f"{avg_fitness:.4f}",
                    "średni_wynik_najlepszego": f"{best.avg_score:.2f}",
                    "średni_wynik_populacji": f"{avg_score:.2f}",
                    "najlepszy_wynik_najlepszego": best.best_score,
                    "średnie_kroki_najlepszego": f"{best.avg_steps:.2f}",
                    "liczba_genomów": len(scored_genomes),
                }
            )
            file.flush()

            print(
                f"pokolenie {generation['number']}/{GENERATIONS}, "
                f"najlepsza sprawność {best.fitness:.2f}, "
                f"średni wynik najlepszego {best.avg_score:.2f}"
            )

        winner = population.run(eval_genomes, GENERATIONS)

    temp_history_path.replace(history_path)

    with open(NEAT_MODEL_FILE, "wb") as file:
        pickle.dump(winner, file)

    print(f"Model NEAT zapisany do {NEAT_MODEL_FILE}")
    print(f"Historia treningu zapisana do {TRAINING_HISTORY_FILE}")

    return NEATAgent.from_genome(winner, config)


def test(agent):
    env = SnakeRLEnvironment(width=WIDTH, height=HEIGHT, max_steps=MAX_STEPS)
    scores = []
    steps = []

    for _ in range(TEST_GAMES):
        state = env.reset()

        while not env.game.game_over:
            if hasattr(agent, "choose_action_from_env"):
                action = agent.choose_action_from_env(env)
            else:
                action = agent.choose_action(state)
            state, reward, game_over = env.step(action)

        scores.append(env.game.score)
        steps.append(env.game.steps)

    avg_score = sum(scores) / len(scores)
    avg_steps = sum(steps) / len(steps)

    print()
    print("=" * 50)
    print("Test NEAT")
    print("=" * 50)
    print(f"  średni wynik: {avg_score:.2f}")
    print(f"  najlepszy wynik: {max(scores)}")
    print(f"  najgorszy wynik: {min(scores)}")
    print(f"  średnia liczba kroków: {avg_steps:.2f}")
    print(f"  liczba testów: {TEST_GAMES}")
    print("=" * 50)

    return {
        "avg_score": avg_score,
        "best_score": max(scores),
        "avg_steps": avg_steps,
        "test_games": TEST_GAMES,
    }


def main():
    agent = train()
    test(agent)


if __name__ == "__main__":
    main()
