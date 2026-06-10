# Snake RL

Projekt porównuje pięciu agentów grających w Snake'a: losowego, heurystycznego, Q-learning, DQN i NEAT.

## Instalacja

```bash
pip install -r requirements.txt
```

## Uruchamianie

```bash
python display_grid.py
python run_experiments.py
python plot_results.py
```

Najważniejsze pliki podczas prezentacji:

- `display_grid.py` - pokaz agentów,
- `run_experiments.py` - porównanie wyników,
- `plot_results.py` - generowanie wykresów.


Trening lub kontynuacja modeli:

```bash
python -m project.training.train_q_learning
python -m project.training.train_dqn
python -m project.training.train_neat
```

## Wyniki

Plansza `10x10`, po `100` gier testowych. Test nie ma limitu kroków i trwa do końca gry.

| Agent | Średni wynik | Najlepszy wynik | Średnie kroki |
| --- | ---: | ---: | ---: |
| losowy | 0,12 | 2 | 14,14 |
| heurystyka | 20,65 | 41 | 169,81 |
| Q-learning | 16,86 | 31 | 139,29 |
| DQN | 17,97 | 34 | 141,60 |
| NEAT | 59,40 | 97 | 1531,47 |

Wynik `97` oznacza zapełnienie całej planszy: wąż zaczyna z długością 3, więc na planszy mającej 100 pól może zjeść maksymalnie 97 pokarmów.
