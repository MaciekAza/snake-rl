# snake-rl

Prosty projekt: agent gra w Snake'a.

## Uruchamianie

Gra manualna:

```bash
python main.py
```

Baseline:

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

Eksperymenty:

```bash
python run_experiments.py
```

## Q-learning

Dodany jest pierwszy algorytm RL:

- tablica Q,
- epsilon-greedy,
- trening przez 10000 epizodow,
- trening od pustej Q-table,
- zapis Q-table do `q_table.pkl`,
- zapis historii treningu do `results/q_learning_training.csv`.

## DQN

Dodany jest prosty Deep Q-Network:

- PyTorch,
- siec neuronowa,
- replay memory,
- target network,
- epsilon-greedy,
- trening przez 1500 epizodow,
- zapis modelu do `dqn_model.pth`,
- zapis historii treningu do `results/dqn_training.csv`.

## Proste wyniki

Ustawienia: plansza `10x10`, limit `500` krokow, test `100` gier.

| Agent | Sredni wynik | Najlepszy wynik | Srednia liczba krokow | Epizody treningu |
| --- | ---: | ---: | ---: | ---: |
| random | 0.14 | 2 | 16.06 | 0 |
| food_heuristic | 20.01 | 36 | 161.74 | 0 |
| Q-learning | 14.71 | 29 | 137.83 | 10000 |
| DQN | 19.22 | 38 | 152.90 | 1500 |

Wyniki eksperymentow sa w pliku:

```text
results/experiment_results.csv
```

Historie treningu sa w plikach:

```text
results/q_learning_training.csv
results/dqn_training.csv
```
