# snake-rl

Prosty projekt: agent gra w Snake'a.

## Uruchamianie

Gra manualna:

```bash
python main.py
```

Agenci bazowi:

```bash
python baseline_demo.py
```

Q-learning:

```bash
python train_q_learning.py
```

DQN:

```bash
python train_dqn.py
```

NEAT:

```bash
python train_neat.py
```

Eksperymenty:

```bash
python run_experiments.py
```

## Q-learning

Dodany jest pierwszy algorytm RL:

- tablica Q,
- epsilon-greedy,
- trening przez 10000 epizodów,
- trening od pustej Q-table,
- zapis Q-table do `q_table.pkl`,
- zapis historii treningu do `results/q_learning_training.csv`.

## DQN

Dodany jest prosty Deep Q-Network:

- PyTorch,
- sieć neuronowa,
- replay memory,
- target network,
- epsilon-greedy,
- trening przez 1500 epizodów,
- zapis modelu do `dqn_model.pth`,
- zapis historii treningu do `results/dqn_training.csv`.

## NEAT

Dodana jest neuroewolucja:

- biblioteka `neat-python`,
- sieć neuronowa tworzona przez ewolucję,
- populacja genomów,
- selekcja najlepszego osobnika,
- osobna funkcja sprawności z karą za stagnację i pętle,
- trening przez 35 pokoleń,
- zapis najlepszego genomu do `neat_winner.pkl`,
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
| losowy | 0.13 | 1 | 13.21 | 0 |
| heurystyka_jedzenia | 19.50 | 36 | 157.35 | 0 |
| Q-learning | 14.61 | 27 | 116.49 | 10000 |
| DQN | 17.82 | 31 | 143.32 | 1500 |
| NEAT | 24.70 | 37 | 577.77 | 35 pokoleń |

Wyniki eksperymentów są w pliku:

```text
results/experiment_results.csv
```

Historie treningu są w plikach:

```text
results/q_learning_training.csv
results/dqn_training.csv
results/neat_training.csv
```

Aktualne modele używane przez skrypty:

```text
q_table.pkl
dqn_model.pth
neat_winner.pkl
```

Wspólne ścieżki do tych plików są w `project_paths.py`.
