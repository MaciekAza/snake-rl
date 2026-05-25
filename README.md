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
- trening przez 3000 epizodów,
- zapis modelu do `dqn_model.pth`,
- zapis historii treningu do `results/dqn_training.csv`.

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

Ustawienia: plansza `10x10`, limit `500` kroków, test `100` gier.

| Agent | Średni wynik | Najlepszy wynik | Średnia liczba kroków | Epizody treningu |
| --- | ---: | ---: | ---: | ---: |
| losowy | 0.12 | 2 | 15.99 | 0 |
| heurystyka_jedzenia | 20.14 | 38 | 164.05 | 0 |
| Q-learning | 14.67 | 30 | 116.22 | 10000 |
| DQN | 17.38 | 34 | 143.42 | 3000 |

Wyniki eksperymentów są w pliku:

```text
results/experiment_results.csv
```

Historie treningu są w plikach:

```text
results/q_learning_training.csv
results/dqn_training.csv
```
