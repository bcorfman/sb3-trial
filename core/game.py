import math
import os
from dataclasses import dataclass
from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete


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


def _manhattan_distance(loc1: Coord, loc2: Coord):
    return abs(loc1.row - loc2.row) + abs(loc1.col - loc2.col)


class Moves(Enum):
    SPACE_LEFT = 0
    SPACE_DOWN = 1
    SPACE_RIGHT = 2
    SPACE_UP = 3


class NPuzzle:
    def __init__(self, n: int, init_state=[]):
        self.tile_width = len(str(n))
        self.n = n + 1
        side = int(math.sqrt(self.n))
        self.side = side
        self.directions = {
            Moves.SPACE_LEFT.value: Coord(0, -1),
            Moves.SPACE_DOWN.value: Coord(1, 0),
            Moves.SPACE_RIGHT.value: Coord(0, 1),
            Moves.SPACE_UP.value: Coord(-1, 0),
        }
        self._goal_state = np.array(
            list(range(1, self.n)) + [0], dtype=np.int8
        ).reshape((self.side, self.side))
        self.project_dir = os.path.dirname(os.path.os.path.dirname(__file__))
        self.reset(init_state)

    def __repr__(self):
        return f"Puzzle({self.n - 1}, {list(self.field.flatten())})"

    def __str__(self):
        out = []
        for row in range(self.side):
            lst = []
            for col in range(self.side):
                lst.append(
                    f"{self.field[row][col]:>{self.tile_width}}"
                    if self.field[row][col] > 0
                    else " " * self.tile_width
                )
            out.append(" ".join(lst))
        return "\n".join(out)

    def tile_loc(self, tile_num):
        rows, cols = np.where(self.field == tile_num)
        return Coord(rows[0], cols[0])

    def goal_loc(self, tile_num):
        rows, cols = np.where(self._goal_state == tile_num)
        return Coord(rows[0], cols[0])

    def reset(self, init_state=[]):
        for state in self._starting_configurations(init_state):
            self.field = np.array(state, dtype=np.int8).reshape((self.side, self.side))
            self.space = self.tile_loc(0)

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
            filename = os.path.join(
                self.project_dir, "res", str(self.n - 1) + "_init_nodes.txt"
            )
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
        print(str(self))


class NPuzzleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, n=8, render_mode=None):
        self.n = n
        self.render_mode = render_mode
        self.puzzle = NPuzzle(self.n)
        self.action_space = Discrete(len(Moves))
        self.observation_space = Box(
            low=0,
            high=n,
            shape=(self.puzzle.side, self.puzzle.side),
            dtype=np.int8,
        )
        self.last_reward = 0

    # Gymnasium-required function (and parameters) to reset the environment.
    def reset(self, seed=None, options=None):
        super().reset(
            seed=seed
        )  # gym requires this call to control randomness and reproduce scenarios.

        # Reset the N-Puzzle itself.
        self.puzzle.reset()

        # Construct the observation state.
        obs = self._get_obs()

        # Additional info to return, for debugging or other purposes.
        info = {}

        # Render environment
        if self.render_mode == "human":
            self.render()

        return obs, info

    # Gymnasium-required function (and parameters) to perform an action.
    def step(self, action):
        self.puzzle.move(action)  # take a single action

        # Determine reward and termination
        reward = (
            self._get_reward() - self.last_reward
        )  # difference between this reward and last reward
        terminated = self.puzzle.is_goal_state()
        if terminated:
            print("GOAL")

        # Construct the observation state.
        observation = self._get_obs()

        # Additional info to return, for debugging or other purposes.
        info = {}

        # Render the environment.
        if self.render_mode == "human":
            print(Moves(action))
            self.render()

        # return Gymnasium-required parameters. 4th param (truncated) is
        # not used for this environment.
        return (
            observation,
            reward,
            terminated,
            False,
            info,
        )

    def render(self):
        self.puzzle.render()

    def _sum_tile_distances(self):
        return sum(
            (
                _manhattan_distance(self.puzzle.tile_loc(i), self.puzzle.goal_loc(i))
                for i in range(self.n)
            )
        )

    def _get_obs(self):
        return self.puzzle.field

    def _get_reward(self):
        return 1000 if self.puzzle.is_goal_state() else -1 * self._sum_tile_distances()
