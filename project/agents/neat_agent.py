import pickle
from pathlib import Path

from project.environment import ACTIONS
from project.paths import NEAT_CONFIG_FILE, NEAT_MODEL_FILE


NEAT_INPUT_SIZE = 30


def choose_best_action(outputs):
    return ACTIONS[max(range(len(ACTIONS)), key=lambda index: outputs[index])]


def build_neat_inputs(env):
    return env.get_state()


class NEATAgent:
    name = "NEAT"

    def __init__(self, network):
        self.network = network
        self.input_size = len(getattr(network, "input_nodes", [])) or NEAT_INPUT_SIZE

    def choose_action(self, state):
        inputs = [float(value) for value in state]

        if len(inputs) < self.input_size:
            inputs.extend([0.0] * (self.input_size - len(inputs)))
        else:
            inputs = inputs[:self.input_size]

        outputs = self.network.activate(tuple(inputs))
        return choose_best_action(outputs)

    def choose_action_from_env(self, env):
        inputs = build_neat_inputs(env)
        outputs = self.network.activate(inputs)
        return choose_best_action(outputs)

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

        with Path(model_file).open("rb") as file:
            genome = pickle.load(file)

        print(f"Model NEAT wczytany z {model_file}")
        return cls.from_genome(genome, config)
