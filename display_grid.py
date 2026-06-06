import argparse
import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from agents.baseline import FoodAgent, RandomAgent
from agents.q_learning import QLearningAgent
from project_paths import (
    DQN_MODEL_FILE,
    NEAT_CONFIG_FILE,
    NEAT_MODEL_FILE,
    Q_TABLE_FILE,
)
from rl.evaluation import choose_rl_action
from rl.environment import ACTIONS, SnakeRLEnvironment
from settings import BOARD_HEIGHT, BOARD_WIDTH, DISPLAY_MAX_STEPS
from snake_game.game import SnakeGame


Q_KINDS = {"qlearning", "q", "q-learn"}
DQN_KINDS = {"dqn", "dnn"}
NEAT_KINDS = {"neat", "neuroevolution", "neuro-ewolucja"}
FOOD_KINDS = {"food", "heuristic", "food_heuristic"}

DISPLAY_NAMES = {
    "food": "HEURYSTYKA",
    "heuristic": "HEURYSTYKA",
    "food_heuristic": "HEURYSTYKA",
    "qlearning": "Q-LEARNING",
    "q": "Q-LEARNING",
    "q-learn": "Q-LEARNING",
    "dqn": "DQN",
    "dnn": "DQN",
    "neat": "NEAT",
    "neuroevolution": "NEAT",
    "neuro-ewolucja": "NEAT",
    "random": "LOSOWY",
    "DQNAgent": "DQN",
    "NEATAgent": "NEAT",
    "QLearningAgent": "Q-LEARNING",
    "FoodAgent": "HEURYSTYKA",
    "RandomAgent": "LOSOWY",
    "heurystyka_jedzenia": "HEURYSTYKA",
    "losowy": "LOSOWY",
}

BACKGROUND_COLORS = [
    (40, 40, 40),
    (42, 48, 52),
    (40, 38, 40),
    (38, 44, 40),
]


def display_name(value):
    return DISPLAY_NAMES.get(value, value.upper())


class Slot:
    def __init__(self, kind, width, height, max_steps):
        self.kind = kind.lower()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.agent = None
        self.env = None
        self.game = None
        self.is_rl = False
        self._create()

    def _setup_rl(self):
        self.env = SnakeRLEnvironment(
            width=self.width,
            height=self.height,
            max_steps=self.max_steps,
        )
        state = self.env.reset()
        self.game = self.env.game
        self.is_rl = True
        return state

    def _use_baseline(self, agent):
        self.env = None
        self.agent = agent
        self.game = SnakeGame(width=self.width, height=self.height)
        self.is_rl = False

    def _fallback(self, message):
        print(f"{message} Używam agenta losowego.", file=sys.stderr)
        self._use_baseline(RandomAgent())

    def _create_q_learning(self):
        self._setup_rl()

        if not Q_TABLE_FILE.exists():
            self._fallback("Nie znaleziono tablicy Q.")
            return

        agent = QLearningAgent()

        try:
            if not agent.load(Q_TABLE_FILE):
                raise FileNotFoundError(Q_TABLE_FILE)
        except Exception as error:
            self._fallback(f"Nie udało się wczytać tablicy Q: {error}.")
            return

        agent.epsilon = 0
        self.agent = agent

    def _create_dqn(self):
        state = self._setup_rl()

        if not DQN_MODEL_FILE.exists():
            self._fallback("Nie znaleziono modelu DQN.")
            return

        try:
            from agents.dqn import DQNAgent

            agent = DQNAgent(state_size=len(state), action_size=len(ACTIONS))
            agent.load(DQN_MODEL_FILE)
            agent.epsilon = 0
            self.agent = agent
        except Exception as error:
            self._fallback(f"Nie udało się wczytać modelu DQN: {error}.")

    def _create_neat(self):
        self._setup_rl()

        if not NEAT_MODEL_FILE.exists():
            self._fallback("Nie znaleziono modelu NEAT.")
            return
        if not NEAT_CONFIG_FILE.exists():
            self._fallback("Nie znaleziono konfiguracji NEAT.")
            return

        try:
            from agents.neat_agent import NEATAgent

            self.agent = NEATAgent.load(NEAT_MODEL_FILE, NEAT_CONFIG_FILE)
        except Exception as error:
            self._fallback(f"Nie udało się wczytać modelu NEAT: {error}.")

    def _create(self):
        if self.kind in Q_KINDS:
            self._create_q_learning()
        elif self.kind in DQN_KINDS:
            self._create_dqn()
        elif self.kind in NEAT_KINDS:
            self._create_neat()
        elif self.kind in FOOD_KINDS:
            self._use_baseline(FoodAgent())
        else:
            self._use_baseline(RandomAgent())

    def reset(self):
        if self.is_rl:
            self.env.reset()
            self.game = self.env.game
        else:
            self.game = SnakeGame(width=self.width, height=self.height)

    def step(self):
        if self.game.game_over:
            return

        if self.is_rl:
            state = self.env.get_state()
            action = choose_rl_action(self.agent, self.env, state)
            self.env.step(action)
            self.game = self.env.game
        else:
            self.game.step(self.agent.choose_direction(self.game))

    @property
    def name(self):
        raw_name = getattr(self.agent, "name", self.agent.__class__.__name__)
        return display_name(raw_name)


def parse_layout(layout):
    agents = []

    for part in (item.strip() for item in layout.split(",") if item.strip()):
        if ":" in part:
            kind, count = part.split(":", 1)
            agents.extend([kind.strip()] * int(count))
        else:
            agents.append(part)

    return agents or ["random"]


def synchronize_food(slots):
    if not slots:
        return

    food = slots[0]["slot"].game.food

    for entry in slots:
        slot = entry["slot"]
        slot.game.food = food

        if slot.is_rl:
            slot.env.last_food_distance = slot.env._calculate_food_distance()


def reset_slots(slots):
    for entry in slots:
        entry["slot"].reset()

    synchronize_food(slots)


def create_slots(agent_kinds, rows, width, height, max_steps):
    slots = []

    for col, kind in enumerate(agent_kinds):
        normalized_kind = kind.lower()

        for row in range(rows):
            slots.append(
                {
                    "slot": Slot(
                        normalized_kind,
                        width=width,
                        height=height,
                        max_steps=max_steps,
                    ),
                    "row": row,
                    "col": col,
                }
            )

    synchronize_food(slots)
    return slots


def draw_header(screen, font, agent_kinds, cell_pixel_width, margin, info_height):
    for col, kind in enumerate(agent_kinds):
        min_x = margin + col * (cell_pixel_width + margin)
        center_x = min_x + cell_pixel_width // 2
        label = font.render(display_name(kind), True, (240, 240, 240))
        screen.blit(label, (center_x - label.get_width() // 2, 30))

        if col < len(agent_kinds) - 1:
            separator_x = min_x + cell_pixel_width + margin // 2
            pygame.draw.line(
                screen,
                (68, 68, 68),
                (separator_x, info_height),
                (separator_x, screen.get_height() - margin),
                2,
            )


def draw_slot(screen, font, entry, cell_width, cell_height, cell_size, margin, info_height):
    slot = entry["slot"]
    row = entry["row"]
    col = entry["col"]
    pixel_width = cell_width * cell_size
    pixel_height = cell_height * cell_size
    x0 = margin + col * (pixel_width + margin)
    y0 = margin + row * (pixel_height + margin) + info_height

    def cell_rect(x, y):
        return pygame.Rect(
            x0 + x * cell_size,
            y0 + y * cell_size,
            cell_size - 1,
            cell_size - 1,
        )

    pygame.draw.rect(
        screen,
        BACKGROUND_COLORS[col % len(BACKGROUND_COLORS)],
        (x0, y0, pixel_width, pixel_height),
    )

    for x in range(cell_width):
        for y in range(cell_height):
            pygame.draw.rect(screen, (30, 30, 30), cell_rect(x, y))

    food_x, food_y = slot.game.food
    pygame.draw.rect(screen, (200, 50, 50), cell_rect(food_x, food_y))

    for x, y in slot.game.snake[1:]:
        pygame.draw.rect(screen, (50, 160, 50), cell_rect(x, y))

    head_x, head_y = slot.game.snake[0]
    pygame.draw.rect(screen, (100, 220, 100), cell_rect(head_x, head_y))

    text = font.render(
        f"{slot.name}  wynik:{slot.game.score} kroki:{slot.game.steps}",
        True,
        (220, 220, 220),
    )
    screen.blit(text, (x0 + 4, y0 + 4))

    if slot.game.game_over:
        game_over = font.render("KONIEC GRY", True, (240, 80, 80))
        screen.blit(game_over, (x0 + 4, y0 + 18))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Wyświetla kilka gier Snake w jednej siatce",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="pokaż tę pomoc i zakończ")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--cell-w", type=int, default=BOARD_WIDTH, help="szerokość planszy w polach")
    parser.add_argument("--cell-h", type=int, default=BOARD_HEIGHT, help="wysokość planszy w polach")
    parser.add_argument("--cell-size", type=int, default=40, help="liczba pikseli na jedno pole")
    parser.add_argument("--margin", type=int, default=12, help="odstęp w pikselach między planszami")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument(
        "--layout",
        default="food:1,qlearning:1,dqn:1,neat:1",
        help="lista typ:liczba, np. food:1,qlearning:1,dqn:1,neat:1",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DISPLAY_MAX_STEPS,
        help="limit kroków jednej gry w wizualizacji",
    )
    return parser


def main():
    args = build_parser().parse_args()
    agent_kinds = parse_layout(args.layout)
    columns = len(agent_kinds)

    cell_pixel_width = args.cell_w * args.cell_size
    cell_pixel_height = args.cell_h * args.cell_size
    info_height = 58
    window_width = columns * cell_pixel_width + (columns + 1) * args.margin
    window_height = (
        args.rows * cell_pixel_height
        + (args.rows + 1) * args.margin
        + info_height
    )

    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Porównanie agentów Snake")
    font = pygame.font.SysFont(None, 18)
    small_font = pygame.font.SysFont(None, 14)
    clock = pygame.time.Clock()
    slots = create_slots(
        agent_kinds,
        rows=args.rows,
        width=args.cell_w,
        height=args.cell_h,
        max_steps=args.max_steps,
    )
    fps = args.fps
    paused = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset_slots(slots)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    fps = min(60, fps + 1)
                elif event.key == pygame.K_MINUS:
                    fps = max(1, fps - 1)

        if not paused:
            for entry in slots:
                entry["slot"].step()

        screen.fill((20, 20, 20))
        info = font.render(
            f"Wiersze: {args.rows} Kolumny: {columns} FPS: {fps}  "
            "Spacja=pauza  R=restart  +/-=szybkość  Q=wyjście",
            True,
            (220, 220, 220),
        )
        screen.blit(info, (10, 6))
        draw_header(
            screen,
            font,
            agent_kinds,
            cell_pixel_width,
            args.margin,
            info_height,
        )

        for entry in slots:
            draw_slot(
                screen,
                small_font,
                entry,
                args.cell_w,
                args.cell_h,
                args.cell_size,
                args.margin,
                info_height,
            )

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()
