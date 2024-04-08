import math
import random
from typing import List

import gymnasium as gym
import pygame
from gymnasium.error import DependencyNotInstalled
from gymnasium.spaces import Discrete


class Actions:
    SPACE_LEFT = 0
    SPACE_DOWN = 1
    SPACE_RIGHT = 2
    SPACE_UP = 3


class NPuzzle:
    def __init__(self, n: int, init_state=None):
        self.n = n + 1
        side = math.sqrt(n + 1)
        assert side * side == n + 1
        self.side = side
        if init_state:
            assert list(sorted(init_state)) == range(self.n)
        self.field = (
            self.generate_random_puzzle(n) if init_state is None else init_state
        )
        self.space = self.field.index(0)
        self.directions = {
            Actions.SPACE_LEFT: -1,
            Actions.SPACE_DOWN: side,
            Actions.SPACE_RIGHT: 1,
            Actions.SPACE_UP: -side,
        }

    def __repr__(self):
        out = ""
        for row in range(self.side):
            out += f"{self.field[row * self.side] if self.field[row * self.side] > 0 else " "} "
            out += f"{self.field[row * self.side + 1 if self.field[row * self.side+1] > 0 else " "]} "
            out += f"{self.field[row * self.side + 2] if self.field[row * self.side+ 2] > 0 else " "}\n"
        return out

    def move(self, action):
        direction = self.directions[action]
        move_allowed = self.is_in_bounds(direction)
        if move_allowed:
            new_pos = self.space + self.direction
            self.field[self.space], self.field[new_pos] = (
                self.field[new_pos],
                self.field[self.space],
            )
        return move_allowed

    def is_in_bounds(self, direction):
        pos = direction + self.space
        return pos >= 0 and pos < len(self.field)

    def is_goal_state(self):
        return self.state == range(self.n)

    def generate_random_puzzle(self, n: int = 8) -> List[int]:
        return random.shuffle(range(n + 1))

    @property
    def State(self):
        return self.field


class PuzzleEnv(gym.Env):
    def __init__(self):
        self.n = 8
        self.puzzle = NPuzzle(self.n)
        self.action_space = Discrete(4)
        self.observation_space = Discrete(self.n + 1)
        self.done = False
        self.tick_rate = 30

    def reset(self, **kwargs):
        self.done = False
        self.puzzle = NPuzzle(self.n)
        return self.__get_observation__(), ()

    def step(self, action):
        self.done = False
        if self.render_mode == "human":
            self.render()
        self.puzzle.move(action)
        step_reward = self.__calculate_reward__()
        if self.board.is_goal_state():
            self.done = True

        if self.render_mode == "human":
            print(self.__get_observation__())

        observation = self.__get__observation()

        return observation, step_reward, self.done, False, {"Step Reward": step_reward}

    def __calculate_reward__(self):
        return 1 if self.puzzle.is_goal_state() else 0

    def __render__(self):
        if self.render_mode == "human":
            self.screen.fill((0, 0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.tick_rate)
        elif self.render_mode == "terminal":
            return self.__get__observation()

    def render(self, render_mode="none"):
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.clock = pygame.time.Clock()
            self.screen_width = self.screen_height = self.puzzle.side * 30
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            self.screen.fill((0, 0, 0))
            pygame.display.set_caption("Eight Puzzle")

    def __get_observation__(self):
        return self.board.field

    def close(self):
        pygame.display.quit()
        pygame.quit()
