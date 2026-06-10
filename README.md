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

Całe zaplecze projektu znajduje się w jednym katalogu `project/`.

Trening lub kontynuacja modeli:

```bash
python -m project.training.train_q_learning
python -m project.training.train_dqn
python -m project.training.train_neat
```

## Wyniki

Plansza `10x10`, limit `1000` kroków, po `100` gier testowych.

| Agent | Średni wynik | Najlepszy wynik | Średnie kroki |
| --- | ---: | ---: | ---: |
| losowy | 0,12 | 2 | 14,14 |
| heurystyka | 20,65 | 41 | 169,81 |
| Q-learning | 16,86 | 31 | 139,29 |
| DQN | 17,97 | 34 | 141,60 |
| NEAT | 38,69 | 54 | 936,99 |

Pełny opis eksperymentów, wykresy i wnioski znajdują się w [RAPORT.md](RAPORT.md).
