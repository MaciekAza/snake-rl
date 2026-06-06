# snake-rl

Prosty projekt: agent gra w Snake'a.

## Uruchamianie

Agenci bazowi:

```bash
python baseline_demo.py
```

Porównanie agentów w oknie:

```bash
python display_grid.py
```

Q-learning:

```bash
python -m training.train_q_learning
```

DQN:

```bash
python -m training.train_dqn
```

NEAT:

```bash
python -m training.train_neat
```

Eksperymenty:

```bash
python run_experiments.py
```

Wykresy:

```bash
python plot_results.py
```

## Q-learning

Dodany jest pierwszy algorytm RL:

- tablica Q,
- epsilon-greedy,
- trening przez 30000 epizodów,
- możliwość kontynuacji zapisanej Q-table,
- zapis Q-table do `models/q_table.pkl`,
- zapis historii treningu do `results/q_learning_training.csv`.

## DQN

Dodany jest prosty Deep Q-Network:

- PyTorch,
- sieć neuronowa,
- replay memory,
- target network,
- epsilon-greedy,
- trening przez 1500 epizodów,
- zapis modelu do `models/dqn_model.pth`,
- zapis historii treningu do `results/dqn_training.csv`.

## NEAT

Dodana jest neuroewolucja:

- biblioteka `neat-python`,
- sieć neuronowa tworzona przez ewolucję,
- populacja genomów,
- selekcja najlepszego osobnika,
- osobna funkcja sprawności z karą za stagnację i pętle,
- trening przez 80 pokoleń,
- checkpointy do wznawiania długiego treningu,
- zapis najlepszego genomu do `models/neat_winner.pkl`,
- zapis historii treningu do `results/neat_training.csv`.

## Nagrody

Środowisko karze agenta za ryzykowne ruchy:

- jedzenie: `+10`,
- kolizja ze ścianą: `-30`,
- kolizja z ciałem: `-30`,
- limit kroków: `-15`,
- ruch w stronę jedzenia: mały bonus,
- ruch od jedzenia: mała kara,
- bliskość ciała/ściany: stopniowa kara,
- wejście w ciasny obszar: dodatkowa kara.

## Proste wyniki

Ustawienia: plansza `10x10`, limit `1000` kroków, test `100` gier.

| Agent | Średni wynik | Najlepszy wynik | Średnia liczba kroków | Epizody treningu |
| --- | ---: | ---: | ---: | ---: |
| losowy | 0.16 | 2 | 16.14 | 0 |
| heurystyka_jedzenia | 20.03 | 38 | 163.98 | 0 |
| Q-learning | 16.58 | 30 | 130.45 | 30000 |
| DQN | 18.62 | 36 | 152.52 | 1500 |
| NEAT | 41.20 | 55 | 968.26 | 80 pokoleń |

Wyniki eksperymentów są w pliku:

```text
results/experiment_results.csv
```

Wykresy są w katalogu:

```text
results/plots
```

Wygenerowane wykresy:

```text
results/plots/q_learning_training.png
results/plots/dqn_training.png
results/plots/neat_training.png
results/plots/agent_scores.png
results/plots/agent_steps.png
```

Historie treningu są w plikach:

```text
results/q_learning_training.csv
results/dqn_training.csv
results/neat_training.csv
```

Aktualne modele używane przez skrypty:

```text
models/q_table.pkl
models/dqn_model.pth
models/neat_winner.pkl
```

Konfiguracja NEAT:

```text
config/neat_config.txt
```

Checkpointy NEAT:

```text
checkpoints
```

Wymagania z PDF:

```text
docs/projekty2025-26-letni.pdf
```

Wspólne ścieżki do tych plików są w `project_paths.py`.
Wspólne ustawienia planszy, testów i liczby epizodów są w `settings.py`.
