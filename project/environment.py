from collections import deque

from project.game import DOWN, LEFT, RIGHT, UP, SnakeGame


STRAIGHT = "STRAIGHT"
TURN_RIGHT = "TURN_RIGHT"
TURN_LEFT = "TURN_LEFT"

ACTIONS = [STRAIGHT, TURN_RIGHT, TURN_LEFT]
DIRECTION_ORDER = [UP, RIGHT, DOWN, LEFT]


class SnakeRLEnvironment:
    def __init__(
        self,
        width=10,
        height=10,
        max_steps=1000,
        body_collision_penalty=-30.0,
        wall_collision_penalty=-30.0,
        step_limit_penalty=-15.0,
        seed=None,
    ):
        self.game = SnakeGame(width=width, height=height, seed=seed)
        self.max_steps = max_steps
        self.last_food_distance = 0
        self.steps_without_food = 0
        self.body_collision_penalty = body_collision_penalty
        self.wall_collision_penalty = wall_collision_penalty
        self.step_limit_penalty = step_limit_penalty

    def reset(self):
        self.game.reset()
        self.last_food_distance = self._calculate_food_distance()
        self.steps_without_food = 0
        return self.get_state()

    def step(self, action):
        old_score = self.game.score
        old_food_distance = self.last_food_distance
        direction = self.action_to_direction(action)
        next_position = self.game.next_position(direction)

        reward = -0.01
        reward += self._body_proximity_penalty(next_position)
        reward += self._wall_proximity_penalty(next_position)
        reward += self._trap_penalty(next_position)

        self.game.step(direction)

        if self.game.game_over and not self.game.won:
            if self.game.reason == "wall":
                reward = self.wall_collision_penalty
            elif self.game.reason == "body":
                reward = self.body_collision_penalty
            else:
                reward = self.step_limit_penalty

            return self.get_state(), reward, self.game.game_over

        self.last_food_distance = self._calculate_food_distance()
        if self.last_food_distance < old_food_distance:
            reward += 0.15
        elif self.last_food_distance > old_food_distance:
            reward -= 0.10
        elif self.steps_without_food > 10:
            reward -= 0.02

        if self.game.score > old_score:
            reward += 10
            self.steps_without_food = 0
        else:
            self.steps_without_food += 1

        if (
            self.max_steps is not None
            and self.game.steps >= self.max_steps
            and not self.game.game_over
        ):
            self.game.game_over = True
            self.game.reason = "limit"
            reward = self.step_limit_penalty
        else:
            reward += 0.001

        return self.get_state(), reward, self.game.game_over

    def _calculate_food_distance(self):
        head_x, head_y = self.game.snake[0]
        food_x, food_y = self.game.food
        return abs(head_x - food_x) + abs(head_y - food_y)

    def _get_wall_distance(self, action):
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
        head_x, head_y = self.game.snake[0]
        tail_x, tail_y = self.game.snake[-1]
        return abs(head_x - tail_x) + abs(head_y - tail_y)

    def _body_proximity_penalty(self, position):
        distance = self._get_body_distance_from_position(position)

        if distance == float("inf") or distance > 3:
            return 0

        penalties = {
            1: -1.20,
            2: -0.45,
            3: -0.15,
        }
        return penalties.get(distance, 0)

    def _wall_proximity_penalty(self, position):
        distance = self._get_wall_distance_from_position(position)

        if distance == float("inf") or distance > 1:
            return 0

        penalties = {
            0: -0.35,
            1: -0.10,
        }
        return penalties.get(distance, 0)

    def _trap_penalty(self, position):
        free_area = self._reachable_area_after_move(position)

        if free_area == 0:
            return 0

        needed_area = len(self.game.snake) + 4

        if position == self.game.food:
            needed_area += 1

        if free_area >= needed_area:
            return 0

        missing_ratio = (needed_area - free_area) / needed_area
        return -2.0 * missing_ratio

    def _get_body_distance_from_position(self, position):
        body = self._get_body_segments_for_position(position)

        if len(body) == 0:
            return float("inf")

        x, y = position
        return min(abs(x - body_x) + abs(y - body_y) for body_x, body_y in body)

    def _get_body_segments_for_position(self, position):
        if position == self.game.food:
            return self.game.snake[2:]

        return self.game.snake[2:-1]

    def _get_wall_distance_from_position(self, position):
        x, y = position

        if self.game.is_wall_collision(position):
            return 0

        return min(x, self.game.width - 1 - x, y, self.game.height - 1 - y)

    def _reachable_area_after_move(self, position):
        if self.game.is_wall_collision(position):
            return 0

        eating = position == self.game.food
        blocked = set(self.game.snake if eating else self.game.snake[:-1])

        if position in blocked:
            return 0

        visited = {position}
        queue = deque([position])

        while queue:
            x, y = queue.popleft()

            for next_x, next_y in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                next_position = (next_x, next_y)

                if next_position in visited or next_position in blocked:
                    continue
                if self.game.is_wall_collision(next_position):
                    continue

                visited.add(next_position)
                queue.append(next_position)

        return len(visited)

    def _count_free_neighbors_after_move(self, position):
        if self.game.is_wall_collision(position):
            return 0

        eating = position == self.game.food
        blocked = set(self.game.snake if eating else self.game.snake[:-1])

        if position in blocked:
            return 0

        x, y = position
        neighbors = (
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        )
        return sum(
            not self.game.is_wall_collision(neighbor) and neighbor not in blocked
            for neighbor in neighbors
        )

    def get_state(self):
        head_x, head_y = self.game.snake[0]
        food_x, food_y = self.game.food
        board_area = self.game.width * self.game.height
        max_distance = self.game.width + self.game.height
        current_distance = self._calculate_food_distance()
        state = []

        for action in ACTIONS:
            direction = self.action_to_direction(action)
            next_position = self.game.next_position(direction)
            next_x, next_y = next_position
            next_distance = abs(next_x - food_x) + abs(next_y - food_y)
            body_distance = self._get_body_distance_from_position(next_position)

            if body_distance == float("inf"):
                body_distance = max_distance

            state.extend(
                [
                    self.is_danger(action),
                    (current_distance - next_distance) / max_distance,
                    self._get_wall_distance_from_position(next_position)
                    / max(self.game.width, self.game.height),
                    min(body_distance, max_distance) / max_distance,
                    self._count_free_neighbors_after_move(next_position) / 4,
                    int(next_position == self.game.food),
                ]
            )

        state.extend(
            [
                int(self.game.direction == UP),
                int(self.game.direction == RIGHT),
                int(self.game.direction == DOWN),
                int(self.game.direction == LEFT),
                (food_x - head_x) / max(1, self.game.width - 1),
                (food_y - head_y) / max(1, self.game.height - 1),
                int(food_x < head_x),
                int(food_x > head_x),
                int(food_y < head_y),
                int(food_y > head_y),
                len(self.game.snake) / board_area,
                current_distance / max_distance,
            ]
        )
        return tuple(state)

    def get_q_state(self):
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
            min(self._get_wall_distance(STRAIGHT) // 3, 3),
            min(self._get_wall_distance(TURN_RIGHT) // 3, 3),
            min(self._get_wall_distance(TURN_LEFT) // 3, 3),
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
