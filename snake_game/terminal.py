import os
import sys
import time

from snake_game.game import DOWN, LEFT, RIGHT, UP, SnakeGame


KEY_TO_DIRECTION = {
    "w": UP,
    "d": RIGHT,
    "s": DOWN,
    "a": LEFT,
}


def run_terminal_game(width=20, height=12, speed=0.12):
    game = SnakeGame(width=width, height=height)
    keyboard = Keyboard()

    try:
        draw(game)
        while True:
            key = keyboard.read_key()
            if key == "q":
                break
            if key == "r":
                game.reset()
                draw(game)
                time.sleep(speed)
                continue

            direction = KEY_TO_DIRECTION.get(key)
            if not game.game_over:
                game.step(direction)
            draw(game)
            time.sleep(speed)
    finally:
        keyboard.close()


def draw(game):
    os.system("cls" if os.name == "nt" else "clear")
    print("Snake")
    print("Sterowanie: W/A/S/D | reset: R | wyjscie: Q")
    print(f"Wynik: {game.score} | Kroki: {game.steps}")
    print(game.render_text())
    if game.game_over:
        if game.won:
            print("Wygrana! Wcisnij R, aby zagrac ponownie.")
        else:
            print(f"Koniec gry ({format_reason(game.reason)}). Wcisnij R, aby zagrac ponownie.")


def format_reason(reason):
    reasons = {
        "wall": "kolizja ze sciana",
        "body": "kolizja z cialem",
        "win": "wygrana",
    }
    return reasons.get(reason, "koniec")


class Keyboard:
    def __init__(self):
        self._old_settings = None
        self._termios = None

        if os.name == "nt":
            import msvcrt

            self._msvcrt = msvcrt
            return

        import termios
        import tty

        self._msvcrt = None
        self._termios = termios
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def read_key(self):
        if self._msvcrt is not None:
            if not self._msvcrt.kbhit():
                return None
            key = self._msvcrt.getwch()
            return key.lower()

        import select

        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        return sys.stdin.read(1).lower()

    def close(self):
        if self._old_settings is None or self._termios is None:
            return
        self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._old_settings)
