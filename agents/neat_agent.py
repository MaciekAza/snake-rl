import pickle

from project_paths import NEAT_CONFIG_FILE, NEAT_MODEL_FILE
from rl.environment import ACTIONS


NEAT_INPUT_SIZE = 30


def count_free_neighbors_after_move(env, position):
    game = env.game

    if game.is_wall_collision(position):
        return 0

    eating = position == game.food
    blocked = set(game.snake if eating else game.snake[:-1])

    if position in blocked:
        return 0

    free = 0

    for neighbor in (
        (position[0] + 1, position[1]),
        (position[0] - 1, position[1]),
        (position[0], position[1] + 1),
        (position[0], position[1] - 1),
    ):
        if game.is_wall_collision(neighbor):
            continue
        if neighbor in blocked:
            continue

        free += 1

    return free


def build_neat_inputs(env):
    game = env.game
    head_x, head_y = game.snake[0]
    food_x, food_y = game.food
    board_area = game.width * game.height
    max_distance = game.width + game.height
    current_distance = abs(head_x - food_x) + abs(head_y - food_y)

    inputs = []

    for action in ACTIONS:
        direction = env.action_to_direction(action)
        next_position = game.next_position(direction)
        next_x, next_y = next_position
        next_distance = abs(next_x - food_x) + abs(next_y - food_y)
        body_distance = env._get_body_distance_from_position(next_position)

        if body_distance == float("inf"):
            body_distance = max_distance

        inputs.extend(
            [
                env.is_danger(action),
                (current_distance - next_distance) / max_distance,
                env._get_wall_distance_from_position(next_position) / max(game.width, game.height),
                min(body_distance, max_distance) / max_distance,
                count_free_neighbors_after_move(env, next_position) / 4,
                int(next_position == game.food),
            ]
        )

    base_state = env.get_state()
    inputs.extend(
        [
            base_state[3],
            base_state[4],
            base_state[5],
            base_state[6],
            (food_x - head_x) / max(1, game.width - 1),
            (food_y - head_y) / max(1, game.height - 1),
            base_state[7],
            base_state[8],
            base_state[9],
            base_state[10],
            len(game.snake) / board_area,
            current_distance / max_distance,
        ]
    )

    return tuple(inputs)


class NEATAgent:
    def __init__(self, network):
        self.network = network
        self.name = "NEAT"
        self.input_size = len(getattr(network, "input_nodes", [])) or NEAT_INPUT_SIZE

    def choose_action(self, state):
        inputs = list(float(value) for value in state)

        if len(inputs) < self.input_size:
            inputs.extend([0.0] * (self.input_size - len(inputs)))
        elif len(inputs) > self.input_size:
            inputs = inputs[:self.input_size]

        outputs = self.network.activate(tuple(inputs))
        safe_indexes = [index for index, danger in enumerate(state[:len(ACTIONS)]) if not danger]

        if len(safe_indexes) == 0:
            safe_indexes = list(range(len(ACTIONS)))

        best_index = max(safe_indexes, key=lambda index: outputs[index])
        return ACTIONS[best_index]

    def choose_action_from_env(self, env):
        inputs = build_neat_inputs(env)
        outputs = self.network.activate(inputs)
        safe_indexes = [index for index, action in enumerate(ACTIONS) if not env.is_danger(action)]

        if len(safe_indexes) == 0:
            safe_indexes = list(range(len(ACTIONS)))

        best_index = max(safe_indexes, key=lambda index: outputs[index])
        return ACTIONS[best_index]

    @classmethod
    def from_genome(cls, genome, config):
        import neat

        network = neat.nn.FeedForwardNetwork.create(genome, config)
        return cls(network)

    @classmethod
    def load(cls, model_file=NEAT_MODEL_FILE, config_file=NEAT_CONFIG_FILE):
        import neat

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(config_file),
        )

        with open(model_file, "rb") as file:
            genome = pickle.load(file)

        print(f"Model NEAT wczytany z {model_file}")
        return cls.from_genome(genome, config)
