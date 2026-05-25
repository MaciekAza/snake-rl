import random

from snake_game.game import DIRECTIONS


class RandomAgent:
    def __init__(self):
        self.name = "losowy"

    def choose_direction(self, game):
        return random.choice(DIRECTIONS)


class SafeRandomAgent:
    def __init__(self):
        self.name = "bezpieczny_losowy"

    def choose_direction(self, game):
        safe = game.safe_directions()

        if len(safe) == 0:
            return game.direction

        return random.choice(safe)


class FoodAgent:
    def __init__(self):
        self.name = "heurystyka_jedzenia"

    def choose_direction(self, game):
        safe = game.safe_directions()

        if len(safe) == 0:
            return game.direction

        best_direction = safe[0]
        best_distance = self.distance_to_food(game, best_direction)

        for direction in safe:
            distance = self.distance_to_food(game, direction)

            if distance < best_distance:
                best_direction = direction
                best_distance = distance

        return best_direction

    def distance_to_food(self, game, direction):
        next_x, next_y = game.next_position(direction)
        food_x, food_y = game.food

        return abs(next_x - food_x) + abs(next_y - food_y)
