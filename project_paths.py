from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
CONFIG_DIR = PROJECT_DIR / "config"
DOCS_DIR = PROJECT_DIR / "docs"
RESULTS_DIR = PROJECT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"

Q_TABLE_FILE = MODELS_DIR / "q_table.pkl"
DQN_MODEL_FILE = MODELS_DIR / "dqn_model.pth"
NEAT_MODEL_FILE = MODELS_DIR / "neat_winner.pkl"
NEAT_CONFIG_FILE = CONFIG_DIR / "neat_config.txt"
NEAT_CHECKPOINT_PREFIX = CHECKPOINTS_DIR / "neat-"

Q_LEARNING_HISTORY_FILE = RESULTS_DIR / "q_learning_training.csv"
DQN_HISTORY_FILE = RESULTS_DIR / "dqn_training.csv"
NEAT_HISTORY_FILE = RESULTS_DIR / "neat_training.csv"
EXPERIMENT_RESULTS_FILE = RESULTS_DIR / "experiment_results.csv"
