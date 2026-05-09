import random


UP = "UP"
RIGHT = "RIGHT"
DOWN = "DOWN"
LEFT = "LEFT"

MOVES = {
    UP: (0, -1),
    RIGHT: (1, 0),
    DOWN: (0, 1),
    LEFT: (-1, 0),
}

OPPOSITE = {
    UP: DOWN,
    RIGHT: LEFT,
    DOWN: UP,
    LEFT: RIGHT,
}


class SnakeGame:
    def __init__(self, width=20, height=12, seed=None):
        if width < 5 or height < 5:
            raise ValueError("Plansza musi miec co najmniej 5x5 pol.")

        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.reset()

    def reset(self):
        center_x = self.width // 2
        center_y = self.height // 2

        self.snake = [
            (center_x, center_y),
            (center_x - 1, center_y),
            (center_x - 2, center_y),
        ]
        self.food = self.spawn_food()
        self.direction = RIGHT
        self.next_direction = RIGHT
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.won = False
        self.reason = None

        return self.get_state()

    def step(self, direction=None):
        if self.game_over:
            return self.get_state()

        if direction is not None:
            self.change_direction(direction)

        self.direction = self.next_direction

        move_x, move_y = MOVES[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + move_x, head_y + move_y)
        self.steps += 1

        if self.is_wall_collision(new_head):
            self.game_over = True
            self.reason = "wall"
            return self.get_state()

        ate_food = new_head == self.food
        body = self.snake if ate_food else self.snake[:-1]

        if new_head in body:
            self.game_over = True
            self.reason = "body"
            return self.get_state()

        self.snake.insert(0, new_head)

        if ate_food:
            self.score += 1
            if len(self.snake) == self.width * self.height:
                self.game_over = True
                self.won = True
                self.reason = "win"
            else:
                self.food = self.spawn_food()
        else:
            self.snake.pop()

        return self.get_state()

    def change_direction(self, direction):
        if direction not in MOVES:
            return
        if direction == OPPOSITE[self.direction]:
            return
        self.next_direction = direction

    def get_state(self):
        return {
            "width": self.width,
            "height": self.height,
            "snake": list(self.snake),
            "food": self.food,
            "direction": self.direction,
            "score": self.score,
            "steps": self.steps,
            "game_over": self.game_over,
            "won": self.won,
            "reason": self.reason,
        }

    def render_text(self):
        board = []
        for _ in range(self.height):
            board.append([" " for _ in range(self.width)])

        food_x, food_y = self.food
        board[food_y][food_x] = "*"

        for x, y in self.snake[1:]:
            board[y][x] = "o"

        head_x, head_y = self.snake[0]
        board[head_y][head_x] = "@"

        border = "#" * (self.width + 2)
        lines = [border]
        for row in board:
            lines.append("#" + "".join(row) + "#")
        lines.append(border)

        return "\n".join(lines)

    def is_wall_collision(self, position):
        x, y = position
        return x < 0 or x >= self.width or y < 0 or y >= self.height

    def spawn_food(self):
        free_places = []

        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in self.snake:
                    free_places.append((x, y))

        return self.random.choice(free_places)
