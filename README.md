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

## Q-learning

Dodany jest pierwszy algorytm RL:

- tablica Q,
- epsilon-greedy,
- trening przez 10000 epizodow,
- zapis Q-table do `q_table.pkl`,
- zapis historii treningu do `results/q_learning_training.csv`.

## Proste wyniki

Ustawienia: plansza `10x10`, limit `500` krokow, test `100` gier.

| Agent | Sredni wynik | Najlepszy wynik | Srednia liczba krokow |
| --- | ---: | ---: | ---: |
| random | 0.14 | 2 | 14.70 |
| safe_random | 3.25 | 7 | 414.64 |
| food_heuristic | 20.12 | 34 | 163.23 |
| Q-learning | 15.86 | 34 | 134.71 |

Historia treningu Q-learningu jest w pliku:

```text
results/q_learning_training.csv
```
