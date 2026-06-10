import csv
import pickle
import random
from pathlib import Path
from statistics import fmean

try:
    import neat
except ModuleNotFoundError:
    print("Brakuje biblioteki neat-python. Uruchom: pip install -r requirements.txt")
    raise SystemExit(1)

from project.agents.neat_agent import NEATAgent, build_neat_inputs, choose_best_action
from project.environment import ACTIONS, SnakeRLEnvironment
from project.evaluation import evaluate_rl, print_evaluation
from project.paths import NEAT_CHECKPOINT_PREFIX, NEAT_CONFIG_FILE, NEAT_HISTORY_FILE, NEAT_MODEL_FILE
from project.settings import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    MAX_STEPS,
    NEAT_GENERATIONS,
    RANDOM_SEED,
    TEST_GAMES,
)


EVALUATION_SEEDS = (101, 202, 303, 404, 505)
MAX_STEPS_WITHOUT_FOOD = 80
CHECKPOINT_EVERY_GENERATIONS = 5
RESUME_FROM_CHECKPOINT = True
HISTORY_FIELDS = [
    "pokolenie",
    "najlepsza_sprawność",
    "średnia_sprawność",
    "średni_wynik_najlepszego",
    "średni_wynik_populacji",
    "najlepszy_wynik_najlepszego",
    "średnie_kroki_najlepszego",
    "liczba_genomów",
]


def make_config():
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(NEAT_CONFIG_FILE),
    )


def get_latest_checkpoint():
    prefix_path = Path(NEAT_CHECKPOINT_PREFIX)
    checkpoint_dir = prefix_path.parent

    if not checkpoint_dir.exists():
        return None, 0

    checkpoints = []

    for path in checkpoint_dir.glob(f"{prefix_path.name}*"):
        suffix = path.name.removeprefix(prefix_path.name)

        try:
            generation = int(suffix)
        except ValueError:
            continue

        checkpoints.append((generation, path))

    if len(checkpoints) == 0:
        return None, 0

    generation, path = max(checkpoints, key=lambda item: item[0])
    return path, generation


def make_population(config):
    if RESUME_FROM_CHECKPOINT:
        checkpoint_path, checkpoint_generation = get_latest_checkpoint()

        if checkpoint_path is not None:
            print(f"Wznawiam NEAT z checkpointu {checkpoint_path}")
            return neat.Checkpointer.restore_checkpoint(str(checkpoint_path)), checkpoint_generation

    return neat.Population(config), 0


def trim_history_to_generation(history_path, generation):
    if generation <= 0 or not history_path.exists():
        return

    text = history_path.read_text(encoding="utf-8").replace("\x00", "")
    reader = csv.DictReader(text.splitlines())
    fieldnames = reader.fieldnames
    rows = []

    for row in reader:
        try:
            row_generation = int(row["pokolenie"])
        except (KeyError, ValueError):
            continue

        if row_generation <= generation:
            rows.append(row)

    if fieldnames is None:
        return

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_action(network, env):
    outputs = network.activate(build_neat_inputs(env))
    safe_indexes = [index for index, action in enumerate(ACTIONS) if not env.is_danger(action)]
    return choose_best_action(outputs, safe_indexes)


def play_game(network, seed=None):
    env = SnakeRLEnvironment(
        width=BOARD_WIDTH,
        height=BOARD_HEIGHT,
        max_steps=MAX_STEPS,
        seed=seed,
    )
    env.reset()
    fitness = 0
    steps_without_food = 0
    visited_states = {}

    while not env.game.game_over:
        old_score = env.game.score
        old_distance = env._calculate_food_distance()
        action = choose_action(network, env)
        env.step(action)
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

    avg_score = fmean(scores)
    avg_steps = fmean(steps)
    fitness = fmean(fitness_values)

    genome.fitness = fitness
    genome.avg_score = avg_score
    genome.best_score = max(scores)
    genome.avg_steps = avg_steps

    return fitness


def train():
    random.seed(RANDOM_SEED)
    config = make_config()
    population, start_generation = make_population(config)
    config = population.config
    population.add_reporter(neat.StatisticsReporter())
    Path(NEAT_CHECKPOINT_PREFIX).parent.mkdir(parents=True, exist_ok=True)
    population.add_reporter(
        neat.Checkpointer(
            CHECKPOINT_EVERY_GENERATIONS,
            None,
            str(NEAT_CHECKPOINT_PREFIX),
        )
    )

    history_path = Path(NEAT_HISTORY_FILE)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    trim_history_to_generation(history_path, start_generation)
    file_mode = "a" if start_generation > 0 and history_path.exists() else "w"
    generation = {"number": start_generation}
    generations_to_run = max(0, NEAT_GENERATIONS - start_generation)

    if generations_to_run == 0:
        print(f"NEAT ma już {start_generation} pokoleń, nie trzeba trenować dalej.")
        return NEATAgent.load(NEAT_MODEL_FILE, NEAT_CONFIG_FILE)

    with history_path.open(file_mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=HISTORY_FIELDS,
        )
        if file_mode == "w":
            writer.writeheader()

        def eval_genomes(genomes, neat_config):
            generation["number"] += 1

            for genome_id, genome in genomes:
                evaluate_genome(genome, neat_config)

            scored_genomes = [genome for genome_id, genome in genomes]
            best = max(scored_genomes, key=lambda genome: genome.fitness)
            avg_fitness = fmean(genome.fitness for genome in scored_genomes)
            avg_score = fmean(genome.avg_score for genome in scored_genomes)

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
                f"pokolenie {generation['number']}/{NEAT_GENERATIONS}, "
                f"najlepsza sprawność {best.fitness:.2f}, "
                f"średni wynik najlepszego {best.avg_score:.2f}"
            )

        winner = population.run(eval_genomes, generations_to_run)

    model_path = Path(NEAT_MODEL_FILE)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as file:
        pickle.dump(winner, file)

    print(f"Model NEAT zapisany do {NEAT_MODEL_FILE}")
    print(f"Historia treningu zapisana do {NEAT_HISTORY_FILE}")

    return NEATAgent.from_genome(winner, config)


def test(agent):
    metrics = evaluate_rl(
        agent,
        games=TEST_GAMES,
        width=BOARD_WIDTH,
        height=BOARD_HEIGHT,
        max_steps=MAX_STEPS,
    )
    print_evaluation("Test NEAT", metrics)
    return metrics


def main():
    agent = train()
    test(agent)


if __name__ == "__main__":
    main()
