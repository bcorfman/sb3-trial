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
        """
        Initialize the N-Puzzle game.

        :param n: The size of the puzzle (e.g., 8 for an 8-puzzle)
        :param init_state: Optional initial state of the puzzle
        """
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
        self.tile_bit_width = 2 * side
        board_bit_width = self.tile_bit_width * self.n
        board_type = np.int64 if board_bit_width == 64 else np.int128
        self.encoding = np.zeros(1, dtype=board_type)
        self.project_dir = os.path.dirname(os.path.os.path.dirname(__file__))
        self.reset(init_state)

    def __repr__(self):
        """
        :return: a printable representation of the Puzzle
        """
        return f"Puzzle({self.n - 1}, {list(self.field.flatten())})"

    def __str__(self):
        """
        :return: Formatted string representation of the puzzle
        """
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
        """
        Get the location of a specific tile in the puzzle.

        :param tile_num: The number of the tile to locate
        :return: Coord object representing the location of the tile
        """
        rows, cols = np.where(self.field == tile_num)
        return Coord(rows[0], cols[0])

    def goal_loc(self, tile_num):
        """
        Get the goal location of a specific tile in the puzzle.

        :param tile_num: The number of the tile to locate
        :return: Coord object representing the goal location of the tile
        """
        rows, cols = np.where(self._goal_state == tile_num)
        return Coord(rows[0], cols[0])

    def reset(self, init_states=[]):
        """
        Reset the puzzle to its initial state.

        :param init_states: Optional list of initial states to choose from
        """
        for state in self._starting_configurations(init_states):
            if not self.is_solvable(state):
                raise ValueError("Puzzle not solvable")
            self.field = np.array(state, dtype=np.int8).reshape((self.side, self.side))
            self.space = self.tile_loc(0)

    def _count_inversions(self, state):
        """
        Counts the number of inversions in the puzzle.
        An inversion occurs when a tile precedes another tile with a lower number.

        :param state: The puzzle state represented as a list
        :return: The number of inversions in the puzzle state
        """
        inversions = 0
        non_zero_tiles = [tile for tile in state if tile > 0]
        for i in range(len(non_zero_tiles)):
            for j in range(i + 1, len(non_zero_tiles)):
                if non_zero_tiles[i] > non_zero_tiles[j]:
                    inversions += 1
        return inversions

    def is_solvable(self, state):
        """
        Determines if the N-puzzle game is solvable based on the inversion count and blank tile position.

        :param state: The puzzle state represented as a list
        :return: True if the puzzle is solvable, False otherwise
        """
        inversions = self._count_inversions(state)
        blank_row = state.index(0)  # a blank is represented by a 0.
        size = len(state)

        if size % 2 == 1:
            # For odd-sized puzzles, the number of inversions should be even for the puzzle to be solvable.
            return inversions % 2 == 0
        else:
            # For even-sized puzzles, the number of inversions should be odd if the blank tile is on an even row (counting from the bottom),
            # and even if the blank tile is on an odd row.
            return (blank_row % 2 == 0) == (inversions % 2 == 1)

    def move(self, action):
        """
        Perform a move action on the puzzle.

        :param action: The move action to perform (e.g., Moves.SPACE_LEFT)
        :return: True if the move is allowed, False otherwise
        """
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
        """
        Check if a move in a given direction is within the puzzle bounds.

        :param direction: The direction to check (e.g., Coord(0, -1) for left)
        :return: True if the move is within bounds, False otherwise
        """
        new_loc = self.space + direction
        row, col = new_loc.as_tuple()
        return 0 <= row < self.side and 0 <= col < self.side

    def is_goal_state(self):
        """
        Check if the current state of the puzzle is the goal state.

        :return: True if the current state is the goal state, False otherwise
        """
        return (self.field == self._goal_state).all()

    def _starting_configurations(self, init_state=[]):
        """
        Generate starting configurations for the puzzle.

        :param init_state: Optional initial state to use
        :return: Generator yielding starting configurations
        """
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
        """Render the current state of the puzzle."""
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

    def reset(self, seed=None, options=None):
        """Gymnasium-required function (and parameters) to reset the environment."""
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

    def step(self, action):
        """Gymnasium-required function (and parameters) to perform an action."""
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
