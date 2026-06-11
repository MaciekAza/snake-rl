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

Plansza `10x10`, po `100` gier testowych. Q-learning korzysta z prostszego stanu `15` cech, a DQN i NEAT ze stanu `30` cech.

Test nie ma limitu kroków. Kończy się po kolizji, zapełnieniu planszy albo wykryciu dokładnie powtarzającego się układu, który dla deterministycznego agenta oznacza nieskończoną pętlę.

| Agent | Średni wynik | Najlepszy wynik | Średnie kroki |
| --- | ---: | ---: | ---: |
| losowy | 0,12 | 2 | 14,14 |
| heurystyka | 20,65 | 41 | 169,81 |
| Q-learning | 16,86 | 31 | 139,29 |
| DQN | 23,37 | 36 | 220,20 |
| NEAT | 59,40 | 97 | 1531,47 |

Wynik `97` oznacza zapełnienie całej planszy: wąż zaczyna z długością 3, więc na planszy mającej 100 pól może zjeść maksymalnie 97 pokarmów.

Bogatszy stan pomógł DQN, którego średni wynik wzrósł z `17,97` do `23,37`. Dla tablicowego Q-learningu pozostawiono prostsze 15 cech, ponieważ przy 30 cechach liczba oddzielnych stanów była zbyt duża.
