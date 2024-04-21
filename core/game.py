import math
import os
import random
from dataclasses import dataclass
from enum import Enum

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete

BLACK = (0, 0, 0)
LIGHT_GRAY = (128, 128, 128)
WHITE = (255, 255, 255)
LINE_WIDTH = 3


def _manhattan_distance(row1, col1, row2, col2):
    return abs(row1 - row2) + abs(col1 - col2)


@dataclass(eq=False)
class Coord:
    row: int
    col: int

    def __eq__(self, other):
        if isinstance(other, Coord):
            return self.row == other.row and self.col == other.col
        return self.row == other[0] and self.col == other[1]

    def __add__(self, other):
        if isinstance(other, Coord):
            return Coord(self.row + other.row, self.col + other.col)
        else:
            return Coord(self.row + other[0], self.col + other[1])

    def __radd__(self, other):
        if isinstance(other, Coord):
            return self.__add__(other)
        else:
            return Coord(self.row + other[0], self.col + other[1])

    def __hash__(self):
        return hash((self.row, self.col))

    def as_tuple(self):
        return self.row, self.col

    @classmethod
    def from_tuple(cls, t):
        coord = cls.__new__(cls)
        coord.row, coord.col = t
        return coord


class Actions(Enum):
    SPACE_LEFT = 0
    SPACE_DOWN = 1
    SPACE_RIGHT = 2
    SPACE_UP = 3


class NPuzzle:
    def __init__(self, n: int, init_state=[]):
        self.n = n + 1
        side = int(math.sqrt(self.n))
        self.side = side
        self.directions = {
            Actions.SPACE_LEFT: Coord(0, -1),
            Actions.SPACE_DOWN: Coord(1, 0),
            Actions.SPACE_RIGHT: Coord(0, 1),
            Actions.SPACE_UP: Coord(-1, 0),
        }
        self._goal_state = np.array(
            list(range(1, self.n)) + [0], dtype=np.int8
        ).reshape((self.side, self.side))
        self.reset(init_state)

    def __repr__(self):
        return f"Puzzle({self.n - 1}, {list(self.field.flatten())})"

    def __str__(self):
        out = ""
        for row in range(self.side):
            out += " ".join(
                str(self.field[row][col]) if self.field[row][col] > 0 else " "
                for col in range(self.side)
            )
            out += "\n"
        return out

    def reset(self, init_state=[]):
        for state in self._starting_configurations(init_state):
            self.field = np.array(state, dtype=np.int8).reshape((self.side, self.side))
            rows, cols = np.where(self.field == 0)
            self.space = Coord(rows[0], cols[0])

    def move(self, action):
        direction = self.directions[action]
        move_allowed = self.is_in_bounds(direction)
        if move_allowed:
            arr = self.field
            srow, scol = self.space.as_tuple()
            new_loc = self.space + direction
            drow, dcol = new_loc.as_tuple()
            arr[srow][scol], arr[drow][dcol] = arr[drow][dcol], arr[srow][scol]
            self.space.row, self.space.col = drow, dcol
        return move_allowed

    def is_in_bounds(self, direction: Coord):
        new_loc = self.space + direction
        row, col = new_loc.as_tuple()
        return 0 <= row < self.side and 0 <= col < self.side

    def is_goal_state(self):
        return (self.field == self._goal_state).all()

    def _starting_configurations(self, init_state=[]):
        if init_state == []:
            filename = os.path.join("res", str(self.n - 1) + "_init_nodes.txt")
            with open(filename) as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    lst = [int(i) for i in line.strip().split(",")]
                    yield lst
        else:
            yield init_state

    def render(self):
        for row in range(self.side):
            print(" ".join(self.field[row][col] for col in range(self.side)))


class NPuzzleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, n=3, render_mode=None):
        self.n = n
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.puzzle = NPuzzle(self.n)
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=0,
            high=n,
            shape=(self.puzzle.side, self.puzzle.side),
            dtype=np.int8,
        )
        self.done = False
        self.tick_rate = 30
        self.window = None
        self.window_size = 512
        self.cell_size = self.window_size // n
        self.clock = None
        self.tile_sum = 0
        self.tile_font = None

    def reset(self, **kwargs):
        self.puzzle = NPuzzle(self.n)
        self.done = self.puzzle.is_goal_state()
        self.reward = self._get_reward()
        return self._get_obs(), {}

    def step(self, action):
        self.puzzle.move(action)
        observation = self._get_obs()
        return (
            observation,
            self._get_reward(),
            self.puzzle.is_goal_state(),
            False,
            {},
        )

    def render(self):
        pass

    def close(self):
        pygame.display.quit()
        pygame.quit()

    def _sum_tile_distances(self):
        return sum(
            _manhattan_distance(
                self.tile_loc(i) - self.goal_loc(i) for i in range(self.n)
            )
        )

    def _render_frame(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="terminal")'
            )
        else:
            if self.window is None and self.render_mode == "human":
                pygame.init()
                pygame.display.init()
                self.tile_font = self._get_font("freeansbold.ttf", self.window_size)
                self.clock = pygame.time.Clock()
                self.screen_width = self.screen_height = self.puzzle.side * 30
                self.screen = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
                self.screen.fill((0, 0, 0))
                pygame.display.set_caption("Eight Puzzle")
            if self.render_mode == "terminal":
                self._render_frame()
        ###################
        if self.render_mode == "human":
            self.screen.fill(BLACK)
            for i in range(self.n):
                if self.board.State[i] > 0:
                    pygame.draw.rect(
                        self.screen,
                        LIGHT_GRAY,
                        (
                            self.cell_size * i,
                            self.cell_size * i,
                            self.cell_size * (i + 1) - 1,
                            self.cell_size * (i + 1) - 1,
                        ),
                    )
                    tile_number = self.tile_font.render(str(i + 1), True, WHITE)
                    self.screen.blit(
                        tile_number,
                        (
                            self.cell_size * i + tile_number.width // 2,
                            self.cell_size * i + tile_number.height // 2,
                        ),
                    )

            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def _get_obs(self):
        return np.array(self.puzzle.State, dtype=np.int8)

    def _get_reward(self):
        return 1000 if self.done else -1 * self._sum_tile_distances()

    def _get_font(self, path, size):
        project_dir = os.path.dirname(os.path.os.path.dirname(__file__))
        font_path = os.path.join(project_dir, "res", path)
        font = pygame.font.Font(font_path, size)
        return font
