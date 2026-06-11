import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("SNAKE_RL_DATA_DIR", PROJECT_DIR / "data"))
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

Q_TABLE_FILE = MODELS_DIR / "q_table.pkl"
DQN_MODEL_FILE = MODELS_DIR / "dqn_model_30.pth"
NEAT_MODEL_FILE = MODELS_DIR / "neat_winner.pkl"
NEAT_CONFIG_FILE = PROJECT_DIR / "neat_config.txt"
NEAT_CHECKPOINT_PREFIX = CHECKPOINTS_DIR / "neat-"

Q_LEARNING_HISTORY_FILE = RESULTS_DIR / "q_learning_training.csv"
DQN_HISTORY_FILE = RESULTS_DIR / "dqn_training_30.csv"
NEAT_HISTORY_FILE = RESULTS_DIR / "neat_training.csv"
EXPERIMENT_RESULTS_FILE = RESULTS_DIR / "experiment_results.csv"
