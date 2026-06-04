from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

Q_TABLE_FILE = PROJECT_DIR / "q_table.pkl"
DQN_MODEL_FILE = PROJECT_DIR / "dqn_model.pth"
NEAT_MODEL_FILE = PROJECT_DIR / "neat_winner.pkl"
NEAT_CONFIG_FILE = PROJECT_DIR / "neat_config.txt"

Q_LEARNING_HISTORY_FILE = RESULTS_DIR / "q_learning_training.csv"
DQN_HISTORY_FILE = RESULTS_DIR / "dqn_training.csv"
NEAT_HISTORY_FILE = RESULTS_DIR / "neat_training.csv"
EXPERIMENT_RESULTS_FILE = RESULTS_DIR / "experiment_results.csv"
