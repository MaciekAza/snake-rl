import os

from snake_game.game import DOWN, LEFT, RIGHT, UP, SnakeGame


WIDTH = 10
HEIGHT = 10
KEYS = {
    "w": UP,
    "d": RIGHT,
    "s": DOWN,
    "a": LEFT,
}


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    game = SnakeGame(width=WIDTH, height=HEIGHT)
    direction = RIGHT

    while not game.game_over:
        clear_screen()
        print(game.render_text())
        print(f"Wynik: {game.score}  Kroki: {game.steps}")
        print("Sterowanie: W/A/S/D, Enter = ten sam kierunek, Q = wyjście")
        choice = input("Ruch: ").strip().lower()

        if choice == "q":
            print("Przerwano grę.")
            return

        if choice in KEYS:
            direction = KEYS[choice]

        game.step(direction)

    clear_screen()
    print(game.render_text())
    print(f"Koniec gry. Wynik: {game.score}, kroki: {game.steps}")


if __name__ == "__main__":
    main()
