from snake_game.game import DOWN, LEFT, RIGHT, UP, SnakeGame


STRAIGHT = "STRAIGHT"
TURN_RIGHT = "TURN_RIGHT"
TURN_LEFT = "TURN_LEFT"

ACTIONS = [STRAIGHT, TURN_RIGHT, TURN_LEFT]
DIRECTION_ORDER = [UP, RIGHT, DOWN, LEFT]


class SnakeRLEnvironment:
    def __init__(self, width=10, height=10, max_steps=500):
        self.game = SnakeGame(width=width, height=height)
        self.max_steps = max_steps
        self.last_food_distance = 0
        self.steps_without_food = 0

    def reset(self):
        self.game.reset()
        self.last_food_distance = self._calculate_food_distance()
        self.steps_without_food = 0
        return self.get_state()

    def step(self, action):
        old_score = self.game.score
        old_food_distance = self.last_food_distance
        direction = self.action_to_direction(action)

        self.game.step(direction)

        reward = -0.005  # Mniejsza kara za ruch

        # Reward za zbliżanie się do jedzenia
        self.last_food_distance = self._calculate_food_distance()
        if self.last_food_distance < old_food_distance:
            reward += 0.1
        elif self.steps_without_food > 10:
            # Kara za kręcenie się bez zbliżania się do jedzenia
            reward -= 0.05

        # Reward za zdobycie jedzenia
        if self.game.score > old_score:
            reward = 10
            self.steps_without_food = 0
        else:
            self.steps_without_food += 1

        # Kara za kolizję
        if self.game.game_over and not self.game.won:
            reward = -10

        # Limit kroków
        if self.game.steps >= self.max_steps and not self.game.game_over:
            self.game.game_over = True
            self.game.reason = "limit"
            reward = -10
        else:
            # Bonus za przeżycie
            reward += 0.001

        return self.get_state(), reward, self.game.game_over

    def _calculate_food_distance(self):
        """Odległość Manhattanu do jedzenia"""
        head_x, head_y = self.game.snake[0]
        food_x, food_y = self.game.food
        return abs(head_x - food_x) + abs(head_y - food_y)

    def _get_wall_distance(self, action):
        """Odległość do ściany w danym kierunku"""
        direction = self.action_to_direction(action)
        head_x, head_y = self.game.snake[0]
        
        if direction == UP:
            return head_y
        elif direction == DOWN:
            return self.game.height - 1 - head_y
        elif direction == LEFT:
            return head_x
        elif direction == RIGHT:
            return self.game.width - 1 - head_x
        return 0

    def _get_tail_distance(self):
        """Odległość do ogona"""
        head_x, head_y = self.game.snake[0]
        tail_x, tail_y = self.game.snake[-1]
        return abs(head_x - tail_x) + abs(head_y - tail_y)

    def get_state(self):
        head_x, head_y = self.game.snake[0]
        food_x, food_y = self.game.food

        return (
            self.is_danger(STRAIGHT),
            self.is_danger(TURN_RIGHT),
            self.is_danger(TURN_LEFT),
            int(self.game.direction == UP),
            int(self.game.direction == RIGHT),
            int(self.game.direction == DOWN),
            int(self.game.direction == LEFT),
            int(food_x < head_x),
            int(food_x > head_x),
            int(food_y < head_y),
            int(food_y > head_y),
            # Odległości od ścian
            min(self._get_wall_distance(STRAIGHT) // 3, 3),
            min(self._get_wall_distance(TURN_RIGHT) // 3, 3),
            min(self._get_wall_distance(TURN_LEFT) // 3, 3),
            # Odległość od ogona
            min(self._get_tail_distance() // 2, 3),
        )

    def is_danger(self, action):
        direction = self.action_to_direction(action)
        return int(not self.game.is_safe_direction(direction))

    def action_to_direction(self, action):
        index = DIRECTION_ORDER.index(self.game.direction)

        if action == TURN_RIGHT:
            index = (index + 1) % len(DIRECTION_ORDER)
        elif action == TURN_LEFT:
            index = (index - 1) % len(DIRECTION_ORDER)

        return DIRECTION_ORDER[index]
