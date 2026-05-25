import argparse
import sys
import os

import pygame
import pickle
import glob

from snake_game.game import SnakeGame
from rl.environment import SnakeRLEnvironment, ACTIONS
from agents.baseline import FoodAgent, RandomAgent
from agents.q_learning import QLearningAgent


DISPLAY_NAMES = {
    "food": "HEURYSTYKA",
    "heuristic": "HEURYSTYKA",
    "food_heuristic": "HEURYSTYKA",
    "qlearning": "Q-LEARNING",
    "q": "Q-LEARNING",
    "q-learn": "Q-LEARNING",
    "q-learninfg": "Q-LEARNING",
    "dqn": "DQN",
    "dnn": "DQN",
    "random": "LOSOWY",
    "DQNAgent": "DQN",
    "QLearningAgent": "Q-LEARNING",
    "FoodAgent": "HEURYSTYKA",
    "RandomAgent": "LOSOWY",
}


class Slot:
    def __init__(self, kind, width, height, max_steps, agent_filename=None):
        self.kind = kind
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.agent_filename = agent_filename

        self.agent = None
        self.env = None
        self.game = None
        self.is_rl = False
        self.last_food = None

        self._create()

    def _create(self):
        kind = self.kind.lower()

        if kind in ("qlearning", "q", "q-learn", "q-learninfg"):
            self.env = SnakeRLEnvironment(width=self.width, height=self.height, max_steps=self.max_steps)
            state = self.env.reset()
            agent = QLearningAgent()
            try:
                agent.load(self.agent_filename or "q_table.pkl")
            except Exception:
                pass
            agent.epsilon = 0
            self.agent = agent
            self.game = self.env.game
            self.is_rl = True

        elif kind in ("dqn", "dnn"):
            self.env = SnakeRLEnvironment(width=self.width, height=self.height, max_steps=self.max_steps)
            state = self.env.reset()
            state_size = len(state)
            try:
                if self.agent_filename and self.agent_filename.lower().endswith((".pkl", ".pickle")):
                    try:
                        with open(self.agent_filename, "rb") as f:
                            obj = pickle.load(f)

                        if hasattr(obj, "choose_action"):
                            self.agent = obj
                            try:
                                self.agent.epsilon = 0
                            except Exception:
                                pass
                        else:
                            from agents.dqn import DQNAgent as _DQNAgent

                            agent = _DQNAgent(state_size=state_size, action_size=len(ACTIONS))
                            try:
                                agent.load(self.agent_filename)
                            except Exception:
                                pass
                            agent.epsilon = 0
                            self.agent = agent

                    except Exception:
                        from agents.dqn import DQNAgent as _DQNAgent

                        agent = _DQNAgent(state_size=state_size, action_size=len(ACTIONS))
                        try:
                            agent.load(self.agent_filename or "dqn_model.pth")
                        except Exception:
                            pass
                        agent.epsilon = 0
                        self.agent = agent
                else:
                    from agents.dqn import DQNAgent as _DQNAgent

                    agent = _DQNAgent(state_size=state_size, action_size=len(ACTIONS))
                    try:
                        agent.load(self.agent_filename or "dqn_model.pth")
                    except Exception:
                        pass
                    agent.epsilon = 0
                    self.agent = agent
            except Exception:
                print("Ostrzeżenie: DQNAgent jest niedostępny, prawdopodobnie brakuje torch. Używam agenta losowego.", file=sys.stderr)
                self.agent = RandomAgent()

            self.game = self.env.game
            self.is_rl = True

        elif kind in ("food", "heuristic", "food_heuristic"):
            self.game = SnakeGame(width=self.width, height=self.height)
            self.agent = FoodAgent()
            self.is_rl = False

        else:
            self.game = SnakeGame(width=self.width, height=self.height)
            self.agent = RandomAgent()
            self.is_rl = False

    def reset(self):
        if self.is_rl and self.env is not None:
            self.env.reset()
            self.game = self.env.game
        else:
            self.game = SnakeGame(width=self.width, height=self.height)
        try:
            self.last_food = self.game.food
            if self.is_rl:
                self.env.last_food_distance = self.env._calculate_food_distance()
        except Exception:
            self.last_food = None

    def step(self):
        if self.game.game_over:
            return

        if self.is_rl:
            state = self.env.get_state()
            action = self.agent.choose_action(state)
            self.env.step(action)
            self.game = self.env.game
        else:
            direction = self.agent.choose_direction(self.game)
            self.game.step(direction)


def parse_layout(layout_str, total):
    if not layout_str:
        return ["random"] * total

    parts = [p.strip() for p in layout_str.split(",") if p.strip()]
    agents = []

    for part in parts:
        if ":" in part:
            kind, count = part.split(":", 1)
            count = int(count)
            agents.extend([kind.strip()] * count)
        else:
            agents.append(part)

    if len(agents) < total:
        agents.extend(["random"] * (total - len(agents)))
    elif len(agents) > total:
        agents = agents[:total]

    return agents


def main():
    parser = argparse.ArgumentParser(description="Wyświetla kilka gier Snake w jednej siatce")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--cell-w", type=int, default=10, help="szerokość planszy w polach")
    parser.add_argument("--cell-h", type=int, default=10, help="wysokość planszy w polach")
    parser.add_argument("--cell-size", type=int, default=40, help="liczba pikseli na jedno pole")
    parser.add_argument("--margin", type=int, default=12, help="odstęp w pikselach między planszami")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--layout", type=str, default="food:1,qlearning:1,dqn:1",
                        help="lista typ:liczba po przecinkach, np. food:1,qlearning:1,dqn:1")
    parser.add_argument("--dqn-file", type=str, default=None, help="ścieżka do pliku modelu DQN (.pth)")
    parser.add_argument("--qtable-file", type=str, default=None, help="ścieżka do pliku tablicy Q-learning (.pkl)")
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    rows = args.rows
    cols = args.cols
    total = rows * cols
    cell_w = args.cell_w
    cell_h = args.cell_h
    cell_size = args.cell_size
    margin = args.margin
    fps = args.fps

    dqn_file = args.dqn_file
    qtable_file = args.qtable_file

    if not dqn_file:
        pkl_candidates = glob.glob("*dqn*.pkl") + glob.glob("*dqn*.pickle")
        if pkl_candidates:
            dqn_file = pkl_candidates[0]
        else:
            for candidate in ("dqn_model.pth", "dqn_modelt.pth", "dqn_model.pt", "dqn.pkl", "dqn_model.pkl"):
                if os.path.exists(candidate):
                    dqn_file = candidate
                    break

    if not qtable_file:
        if os.path.exists("q_table.pkl"):
            qtable_file = "q_table.pkl"
        else:
            q_candidates = glob.glob("*q_table*.pkl") + glob.glob("*qlearning*.pkl") + glob.glob("*q_learning*.pkl") + glob.glob("*qlearn*.pkl")
            if q_candidates:
                qtable_file = q_candidates[0]

    if dqn_file:
        print(f"Automatycznie wykryto model DQN: {dqn_file}")
    else:
        print("Nie znaleziono modelu DQN. Plansze DQN użyją agenta losowego.")

    if qtable_file:
        print(f"Automatycznie wykryto tablicę Q: {qtable_file}")
    else:
        print("Nie znaleziono tablicy Q. Plansze Q-learning będą działały losowo.")

    agents_kinds = parse_layout(args.layout, total)

    pygame.init()
    pygame.font.init()

    cell_pixel_w = cell_w * cell_size
    cell_pixel_h = cell_h * cell_size

    win_w = cols * cell_pixel_w + (cols + 1) * margin
    info_h = 40
    win_h = rows * cell_pixel_h + (rows + 1) * margin + info_h

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Porównanie agentów Snake")

    font = pygame.font.SysFont(None, 18)
    small_font = pygame.font.SysFont(None, 14)

    zone_kinds = ["food", "qlearning", "dqn"]
    base = cols // 3
    rem = cols % 3
    zone_cols = [base + (1 if i < rem else 0) for i in range(3)]
    zone_starts = []
    s = 0
    for c in zone_cols:
        zone_starts.append(s)
        s += c

    slots = []
    for zone_idx, zone_kind in enumerate(zone_kinds):
        start_col = zone_starts[zone_idx]
        for c in range(zone_cols[zone_idx]):
            col = start_col + c
            for row in range(rows):
                if zone_kind in ("qlearning", "q", "q-learn", "q-learninfg"):
                    filename = qtable_file
                elif zone_kind in ("dqn", "dnn"):
                    filename = dqn_file
                else:
                    filename = None

                slot = Slot(zone_kind, width=cell_w, height=cell_h, max_steps=args.max_steps, agent_filename=filename)
                slots.append({"slot": slot, "row": row, "col": col, "zone": zone_idx})

    if len(slots) > 0:
        master_food = slots[0]["slot"].game.food
        for other in slots:
            s = other["slot"]
            s.game.food = master_food
            s.last_food = master_food
            if s.is_rl:
                s.env.last_food_distance = s.env._calculate_food_distance()

    clock = pygame.time.Clock()
    paused = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    for e in slots:
                        e["slot"].reset()
                    if len(slots) > 0:
                        master_food = slots[0]["slot"].game.food
                        for other in slots:
                            s = other["slot"]
                            s.game.food = master_food
                            s.last_food = master_food
                            if s.is_rl:
                                s.env.last_food_distance = s.env._calculate_food_distance()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    fps = min(60, fps + 1)
                elif event.key == pygame.K_MINUS:
                    fps = max(1, fps - 1)

        if not paused:
            for e in slots:
                e["slot"].step()
                try:
                    e["slot"].last_food = e["slot"].game.food
                except Exception:
                    pass

        screen.fill((20, 20, 20))

        info_surf = font.render(f"Wiersze: {rows} Kolumny: {cols} FPS: {fps}  Spacja=pauza  R=restart  +/-=szybkość  Q=wyjście", True, (220, 220, 220))
        screen.blit(info_surf, (10, 8))

        for zi in range(3):
            zc = zone_cols[zi]
            if zc == 0:
                continue
            cs = zone_starts[zi]
            min_x = margin + cs * (cell_pixel_w + margin)
            zone_width = zc * cell_pixel_w + (zc - 1) * margin
            center_x = int(min_x + zone_width / 2)
            label = DISPLAY_NAMES.get(zone_kinds[zi], zone_kinds[zi].upper())
            label_surf = font.render(label, True, (240, 240, 240))
            screen.blit(label_surf, (center_x - label_surf.get_width() // 2, 8))

            if zi < 2:
                sep_x = int(min_x + zone_width + margin / 2)
                pygame.draw.line(screen, (68, 68, 68), (sep_x, info_h), (sep_x, win_h - margin), 2)

        for entry in slots:
            slot = entry["slot"]
            row = entry["row"]
            col = entry["col"]
            x0 = margin + col * (cell_pixel_w + margin)
            y0 = margin + row * (cell_pixel_h + margin) + info_h

            zone = entry.get("zone", 0)
            bg_colors = [(40, 40, 40), (42, 48, 52), (40, 38, 40)]
            pygame.draw.rect(screen, bg_colors[zone % len(bg_colors)], (x0, y0, cell_pixel_w, cell_pixel_h))

            for x in range(cell_w):
                for y in range(cell_h):
                    rect = pygame.Rect(x0 + x * cell_size, y0 + y * cell_size, cell_size - 1, cell_size - 1)
                    pygame.draw.rect(screen, (30, 30, 30), rect)

            food_x, food_y = slot.game.food
            food_rect = pygame.Rect(x0 + food_x * cell_size, y0 + food_y * cell_size, cell_size - 1, cell_size - 1)
            pygame.draw.rect(screen, (200, 50, 50), food_rect)

            for part in slot.game.snake[1:]:
                px, py = part
                rect = pygame.Rect(x0 + px * cell_size, y0 + py * cell_size, cell_size - 1, cell_size - 1)
                pygame.draw.rect(screen, (50, 160, 50), rect)

            hx, hy = slot.game.snake[0]
            head_rect = pygame.Rect(x0 + hx * cell_size, y0 + hy * cell_size, cell_size - 1, cell_size - 1)
            pygame.draw.rect(screen, (100, 220, 100), head_rect)

            if slot.is_rl:
                raw_name = getattr(slot.agent, "__class__", type(slot.agent)).__name__
            else:
                raw_name = getattr(slot.agent, "name", slot.agent.__class__.__name__)

            name = DISPLAY_NAMES.get(raw_name, raw_name)
            text = f"{name}  wynik:{slot.game.score} kroki:{slot.game.steps}"
            ts = small_font.render(text, True, (220, 220, 220))
            screen.blit(ts, (x0 + 4, y0 + 4))

            if slot.game.game_over:
                over_surf = small_font.render("KONIEC GRY", True, (240, 80, 80))
                screen.blit(over_surf, (x0 + 4, y0 + 18))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()
