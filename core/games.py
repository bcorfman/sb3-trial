import math
import os
import random
from enum import Enum

import numpy as np

from core.util import Coord


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
        board_type = np.int64 if board_bit_width == 64 else np.int32
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


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = 1

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = random.choice([-1, 1])  # Randomly choose the starting player

    @property
    def valid_moves(self):
        moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row, col] == 0:
                    moves.append(row * 3 + col)
        return moves

    def make_move(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.player
            self.player = -self.player
            return True
        return False

    def check_winner(self):
        for i in range(3):
            if np.sum(self.board[i, :]) == 3 or np.sum(self.board[:, i]) == 3:
                print("WIN")
                return 1
            if np.sum(self.board[i, :]) == -3 or np.sum(self.board[:, i]) == -3:
                print("LOSS")
                return -1
        if (
            np.sum(np.diag(self.board)) == 3
            or np.sum(np.diag(np.fliplr(self.board))) == 3
        ):
            print("WIN")
            return 1
        if (
            np.sum(np.diag(self.board)) == -3
            or np.sum(np.diag(np.fliplr(self.board))) == -3
        ):
            print("LOSS")
            return -1
        if np.count_nonzero(self.board) == 9:
            print("TIE")
            return 0
        return None

    def render(self, mode=None):
        if mode == "ansi":
            print(str(self))

    def __str__(self):
        output = "-------------\n"
        for i in range(3):
            output += "| "
            for j in range(3):
                if self.board[i, j] == 1:
                    output += "X | "
                elif self.board[i, j] == -1:
                    output += "O | "
                else:
                    output += "  | "
            output += "\n-------------\n"
        return output
