import csv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from project_paths import (
    DQN_HISTORY_FILE,
    EXPERIMENT_RESULTS_FILE,
    NEAT_HISTORY_FILE,
    PLOTS_DIR,
    Q_LEARNING_HISTORY_FILE,
)


def read_rows(path):
    text = path.read_text(encoding="utf-8").replace("\x00", "")
    return list(csv.DictReader(text.splitlines()))


def numbers(rows, column, cast=float):
    return [cast(row[column]) for row in rows]


def moving_average(values, window):
    if len(values) == 0:
        return []

    result = []
    total = 0

    for index, value in enumerate(values):
        total += value

        if index >= window:
            total -= values[index - window]

        current_window = min(index + 1, window)
        result.append(total / current_window)

    return result


def save_plot(filename):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Zapisano {path}")


def plot_q_learning():
    rows = read_rows(Q_LEARNING_HISTORY_FILE)
    episodes = numbers(rows, "epizod", int)
    scores = numbers(rows, "wynik", int)
    epsilon = numbers(rows, "epsilon")
    average = moving_average(scores, 200)

    fig, axis_score = plt.subplots(figsize=(10, 5))
    axis_score.plot(episodes, scores, color="#b9c7d8", linewidth=0.6, alpha=0.45, label="wynik epizodu")
    axis_score.plot(episodes, average, color="#1f77b4", linewidth=2, label="średnia krocząca 200")
    axis_score.set_title("Q-learning - trening")
    axis_score.set_xlabel("Epizod")
    axis_score.set_ylabel("Wynik")
    axis_score.grid(True, alpha=0.25)

    axis_epsilon = axis_score.twinx()
    axis_epsilon.plot(episodes, epsilon, color="#d62728", linewidth=1.4, alpha=0.85, label="epsilon")
    axis_epsilon.set_ylabel("Epsilon")

    lines, labels = axis_score.get_legend_handles_labels()
    lines_2, labels_2 = axis_epsilon.get_legend_handles_labels()
    axis_score.legend(lines + lines_2, labels + labels_2, loc="upper left")

    save_plot("q_learning_training.png")


def plot_dqn():
    rows = read_rows(DQN_HISTORY_FILE)
    episodes = numbers(rows, "epizod", int)
    scores = numbers(rows, "wynik", int)
    loss = numbers(rows, "średnia_strata")
    average = moving_average(scores, 100)
    loss_average = moving_average(loss, 50)

    fig, axis_score = plt.subplots(figsize=(10, 5))
    axis_score.plot(episodes, scores, color="#c7d8bf", linewidth=0.7, alpha=0.45, label="wynik epizodu")
    axis_score.plot(episodes, average, color="#2ca02c", linewidth=2, label="średnia krocząca 100")
    axis_score.set_title("DQN - trening")
    axis_score.set_xlabel("Epizod")
    axis_score.set_ylabel("Wynik")
    axis_score.grid(True, alpha=0.25)

    axis_loss = axis_score.twinx()
    axis_loss.plot(episodes, loss_average, color="#ff7f0e", linewidth=1.5, alpha=0.9, label="średnia strata 50")
    axis_loss.set_ylabel("Strata")

    lines, labels = axis_score.get_legend_handles_labels()
    lines_2, labels_2 = axis_loss.get_legend_handles_labels()
    axis_score.legend(lines + lines_2, labels + labels_2, loc="upper left")

    save_plot("dqn_training.png")


def plot_neat():
    rows = read_rows(NEAT_HISTORY_FILE)
    generations = numbers(rows, "pokolenie", int)
    best_score = numbers(rows, "średni_wynik_najlepszego")
    population_score = numbers(rows, "średni_wynik_populacji")
    best_fitness = numbers(rows, "najlepsza_sprawność")

    fig, axis_score = plt.subplots(figsize=(10, 5))
    axis_score.plot(generations, best_score, color="#9467bd", linewidth=2.4, marker="o", label="średni wynik najlepszego")
    axis_score.plot(generations, population_score, color="#8c564b", linewidth=1.8, label="średni wynik populacji")
    axis_score.set_title("NEAT - trening")
    axis_score.set_xlabel("Pokolenie")
    axis_score.set_ylabel("Wynik")
    axis_score.grid(True, alpha=0.25)

    axis_fitness = axis_score.twinx()
    axis_fitness.plot(generations, best_fitness, color="#7f7f7f", linewidth=1.2, alpha=0.6, label="najlepsza sprawność")
    axis_fitness.set_ylabel("Sprawność")

    lines, labels = axis_score.get_legend_handles_labels()
    lines_2, labels_2 = axis_fitness.get_legend_handles_labels()
    axis_score.legend(lines + lines_2, labels + labels_2, loc="upper left")

    save_plot("neat_training.png")


def plot_agent_scores():
    rows = read_rows(EXPERIMENT_RESULTS_FILE)
    agents = [row["agent"] for row in rows]
    average_scores = numbers(rows, "średni_wynik")
    best_scores = numbers(rows, "najlepszy_wynik", int)
    indexes = range(len(agents))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar([index - width / 2 for index in indexes], average_scores, width=width, color="#4c78a8", label="średni wynik")
    plt.bar([index + width / 2 for index in indexes], best_scores, width=width, color="#f58518", label="najlepszy wynik")
    plt.title("Porównanie agentów - wyniki")
    plt.xlabel("Agent")
    plt.ylabel("Wynik")
    plt.xticks(list(indexes), agents, rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()

    save_plot("agent_scores.png")


def plot_agent_steps():
    rows = read_rows(EXPERIMENT_RESULTS_FILE)
    agents = [row["agent"] for row in rows]
    steps = numbers(rows, "średnie_kroki")

    plt.figure(figsize=(10, 5))
    plt.bar(agents, steps, color="#54a24b")
    plt.title("Porównanie agentów - średnia liczba kroków")
    plt.xlabel("Agent")
    plt.ylabel("Średnie kroki")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)

    save_plot("agent_steps.png")


def main():
    plot_q_learning()
    plot_dqn()
    plot_neat()
    plot_agent_scores()
    plot_agent_steps()


if __name__ == "__main__":
    main()
