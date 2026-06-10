import random

from project.game import DIRECTIONS


class RandomAgent:
    name = "losowy"

    def choose_direction(self, game):
        return random.choice(DIRECTIONS)


class FoodAgent:
    name = "heurystyka_jedzenia"

    def choose_direction(self, game):
        safe = game.safe_directions()

        if not safe:
            return game.direction

        return min(safe, key=lambda direction: self.distance_to_food(game, direction))

    @staticmethod
    def distance_to_food(game, direction):
        next_x, next_y = game.next_position(direction)
        food_x, food_y = game.food

        return abs(next_x - food_x) + abs(next_y - food_y)
