import math
import os
import random
from typing import List

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete

BLACK = (0, 0, 0)
LIGHT_GRAY = (128, 128, 128)
WHITE = (255, 255, 255)
LINE_WIDTH = 3


class Actions:
    SPACE_LEFT = 0
    SPACE_DOWN = 1
    SPACE_RIGHT = 2
    SPACE_UP = 3


class NPuzzle:
    def __init__(self, n: int, init_state=[]):
        self.n = n + 1
        side = int(math.sqrt(n + 1))
        self.side = side
        self.field = self.generate_random_puzzle(n) if init_state == [] else init_state
        if len(self.field) != self.n:
            raise ValueError("Puzzle init_state is not of length N.")
        self.space = self.field.index(0)
        if self.space < 0:
            raise ValueError("Puzzle does not contain an empty space (0 value)")
        self.directions = {
            Actions.SPACE_LEFT: -1,
            Actions.SPACE_DOWN: side,
            Actions.SPACE_RIGHT: 1,
            Actions.SPACE_UP: -side,
        }
        self.reverse_dir = None

    def __repr__(self):
        return f"Puzzle({self.n - 1}, {self.field})"

    def __str__(self):
        out = ""
        for row in range(self.side):
            out += f"{self.field[row * self.side] if self.field[row * self.side] > 0 else " "} "
            out += f"{self.field[row * self.side + 1] if self.field[row * self.side + 1] > 0 else " "} "
            out += f"{self.field[row * self.side + 2] if self.field[row * self.side + 2] > 0 else " "}\n"
        return out

    def move(self, action):
        direction = self.directions[action]
        move_allowed = self.is_in_bounds(direction) and self.reverse_dir != direction
        if move_allowed:
            self.reverse_dir = -direction
            new_pos = self.space + direction
            self.field[self.space], self.field[new_pos] = (
                self.field[new_pos],
                self.field[self.space],
            )
            self.space = new_pos
        return move_allowed

    def is_in_bounds(self, direction):
        pos = self.space + direction
        pos_row = pos // self.side
        space_row = self.space // self.side
        return (
            pos >= 0
            and pos < len(self.field)
            and (
                (pos_row == space_row and abs(direction) == 1)
                or (pos_row != space_row and abs(direction) == self.side)
            )
        )

    def is_goal_state(self):
        return self.field == list(range(self.n))

    def generate_random_puzzle(self, n: int = 8) -> List[int]:
        puzzle = list(range(n + 1))
        random.shuffle(puzzle)
        return puzzle

    @property
    def State(self):
        state = []
        for row in range(self.side):
            items = []
            for col in range(self.side):
                items.append(self.field[row * self.side + col])
            state.append(items)
        return state


class NPuzzleEnv(gym.Env):
    metadata = {"render_modes": ["human", "terminal"], "render_fps": 4}

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
        self.reward = 0
        self.tile_font = None

    def reset(self, **kwargs):
        self.puzzle = NPuzzle(self.n)
        self.done = self.puzzle.is_goal_state()
        self.reward = self._get_reward()
        return self._get_obs(), {}

    def step(self, action):
        if self.render_mode == "human":
            self._render_frame()
        can_move = self.puzzle.move(action)
        if not can_move:
            self.done = False
        else:
            self.done = self.puzzle.is_goal_state()
        self.reward = self._get_reward()
        observation = self._get_obs()
        if self.render_mode == "terminal":
            print(observation)
        return (
            observation,
            self.reward,
            self.done,
            not can_move,
            {"Step Reward": self.reward},
        )

    def render(self):
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

    def close(self):
        pygame.display.quit()
        pygame.quit()

    def _render_frame(self):
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
        return 100 if self.done else -1

    def _get_font(self, path, size):
        project_dir = os.path.dirname(os.path.os.path.dirname(__file__))
        font_path = os.path.join(project_dir, "res", path)
        font = pygame.font.Font(font_path, size)
        return font
