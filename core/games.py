import math
import os
import random
from collections import Counter
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
        blank_index = state.index(0)  # a blank is represented by a 0.
        blank_row = self.side - blank_index // self.side

        if self.side % 2 == 1:
            # For odd-sized puzzles, the number of inversions should be even for the puzzle to be solvable.
            return inversions % 2 == 0
        else:
            # For even-sized puzzles, the number of inversions should be odd if the blank tile is on an even row (counting from the bottom),
            # and even if the blank tile is on an odd row.
            return (blank_row % 2 == 0 and inversions % 2 == 1) or (
                blank_row % 2 == 1 and inversions % 2 == 0
            )

    def move(self, action):
        """
        Perform a move action on the puzzle.

        :param action: The move action to perform (e.g., Moves.SPACE_LEFT)
        :return: True if the move is allowed, False otherwise
        """
        direction = self.directions[action.value]
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
    EMPTY = 0
    X = 1
    O = 2

    def __init__(self):
        """Initialize the board as a list of 9 elements (all zeros)
        Randomly choose the first player (1 for X, 2 for O)"""
        self.reset()
        self.valid_states = self._map_valid_board_states()

    def __hash__(self):
        """Convert the board state to a binary representation and return it as an unsigned 16-bit integer"""
        return self.valid_states[self.board]

    def __str__(self):
        """Return a string representation of the game board with gridlines"""
        board_str = ""
        for i in range(9):
            if self.board[i] == self.EMPTY:
                board_str += " "
            elif self.board[i] == self.X:
                board_str += "X"
            else:
                board_str += "O"

            if i % 3 == 2:
                board_str += "\n"
            else:
                board_str += "|"

        return board_str

    def reset(self):
        """Reset the board to an empty state and randomly choose the first player"""
        self.board = [self.EMPTY] * 9
        self.player = self.X

    def legal_moves(self, board=None):
        """Return a list of integers representing valid moves based on the current state of the game"""
        if board is None:
            board = self.board

        return [i for i in range(9) if board[i] == self.EMPTY]

    def check_winner(self, player, board=None):
        """Check for a winner or a draw.
        Return 1 if the current player has won, -1 if the opponent has won, 0 for a draw,
        and None if the game is still ongoing."""
        if board is None:
            board = self.board

        winning_combinations = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),  # rows
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),  # columns
            (0, 4, 8),
            (2, 4, 6),  # diagonals
        ]

        for combo in winning_combinations:
            if all(board[i] == player for i in combo):
                return 1
            elif all(board[i] == 3 - player for i in combo):
                return -1

        if board.count(self.EMPTY) == 0:
            return 0

        return None

    def make_move(self, pos):
        """Place an X or O on the board for the current player and then switch players.
        Raise a ValueError if an invalid move is attempted."""
        self.board[pos] = self.player
        self.player = 3 - self.player

    def _map_valid_board_states(self):
        def legal_moves(board):
            return [i for i in range(9) if board[i] == self.EMPTY]

        def move(board, pos, player):
            return board[0:pos] + tuple([player]) + board[pos + 1 :], 3 - player

        board = tuple([self.EMPTY] * 9)
        player = self.X
        frontier = [(board, player)]
        valid_states = set()
        valid_states.add(board)
        while frontier:
            board, player = frontier.pop()
            for pos in legal_moves(board):
                new_board, new_player = move(board, pos, player)
                if new_board not in valid_states:
                    frontier.append((new_board, new_player))
                    valid_states.add(new_board)
        return {state: i for i, state in enumerate(sorted(valid_states))}

    def render(self, mode=None):
        """Called by the TicTacToeEnv class to render the game board."""
        if mode == "ansi":
            print(str(self))
