import io
import sys

import numpy as np

from core.games import Moves, NPuzzle, TicTacToe


def test_create_puzzle_with_init_state():
    init_state = [1, 2, 7, 6, 8, 4, 5, 0, 3]
    puzzle = NPuzzle(8, init_state)
    assert puzzle.side == 3
    assert len(puzzle.field) == 3
    assert (puzzle.field == np.array(init_state).reshape((3, 3))).all()
    allowed = puzzle.move(Moves.SPACE_LEFT)
    assert allowed
    assert (
        puzzle.field == np.array([1, 2, 7, 6, 8, 4, 0, 5, 3]).reshape((puzzle.side, -1))
    ).all()
    allowed = puzzle.move(Moves.SPACE_UP)
    assert allowed
    assert (
        puzzle.field == np.array([1, 2, 7, 0, 8, 4, 6, 5, 3]).reshape((puzzle.side, -1))
    ).all()
    assert puzzle.move(Moves.SPACE_LEFT) == False
    allowed = puzzle.move(Moves.SPACE_RIGHT)
    assert allowed
    assert (
        puzzle.field == np.array([1, 2, 7, 8, 0, 4, 6, 5, 3]).reshape((puzzle.side, -1))
    ).all()
    allowed = puzzle.move(Moves.SPACE_DOWN)
    assert allowed
    assert (
        puzzle.field == np.array([1, 2, 7, 8, 5, 4, 6, 0, 3]).reshape((puzzle.side, -1))
    ).all()


def test_repr():
    # repr(puzzle) returns a string with a
    puzzle = NPuzzle(8, [1, 2, 7, 6, 8, 4, 5, 0, 3])
    assert repr(puzzle) == "Puzzle(8, [1, 2, 7, 6, 8, 4, 5, 0, 3])"


def test_str():
    puzzle = NPuzzle(8, [1, 2, 7, 6, 8, 4, 5, 0, 3])
    assert str(puzzle) == "1 2 7\n6 8 4\n5   3"

    puzzle = NPuzzle(15, [11, 13, 3, 7, 2, 10, 5, 6, 12, 0, 9, 1, 14, 8, 4, 15])
    assert str(puzzle) == "11 13  3  7\n 2 10  5  6\n12     9  1\n14  8  4 15"


def test_goal_state():
    # 8-puzzle goal state is
    # 1 2 3
    # 4 5 6
    # 7 8 0
    puzzle = NPuzzle(8, list(range(1, 9)) + [0])
    assert puzzle.is_goal_state()


def test_starting_configurations():
    # unspecified 8-puzzle starting states draw from valid,
    # precalculated configurations inside a file.
    puzzle = NPuzzle(8)
    start_lst = [
        [3, 4, 8, 6, 2, 5, 0, 1, 7],
        [8, 0, 7, 6, 1, 5, 3, 2, 4],
        [8, 4, 0, 6, 3, 7, 5, 2, 1],
    ]
    for i, lst in enumerate(puzzle._starting_configurations()):
        assert lst == start_lst[i]
        if i == 2:
            break

    # unspecified 15-puzzle starting states draw from valid,
    # precalculated configurations inside a file.
    puzzle = NPuzzle(15)
    start_lst = [
        [2, 5, 6, 1, 4, 8, 14, 12, 9, 10, 13, 11, 0, 3, 15, 7],
        [12, 15, 0, 13, 8, 10, 14, 4, 5, 3, 7, 2, 6, 1, 11, 9],
        [0, 1, 2, 12, 14, 3, 4, 6, 10, 9, 8, 11, 7, 15, 13, 5],
    ]
    for i, lst in enumerate(puzzle._starting_configurations()):
        assert lst == start_lst[i]
        if i == 2:
            break

    # Initial 8-puzzle configurations can also be passed in by the user.
    init_state = list(range(1, 9)) + [0]
    puzzle = NPuzzle(8, init_state)
    for lst in puzzle._starting_configurations(init_state):
        assert lst == init_state


# Should print the current state of the puzzle correctly
def test_print_current_state():
    puzzle = NPuzzle(8, [1, 2, 3, 4, 5, 6, 7, 8, 0])
    captured_output = io.StringIO()
    sys.stdout = captured_output
    puzzle.render()
    sys.stdout = sys.__stdout__
    assert captured_output.getvalue() == "1 2 3\n4 5 6\n7 8  \n"


def test_ttt_number_of_valid_states():
    ttt = TicTacToe()
    assert len(ttt.valid_states) == 6046
    assert ttt.valid_states[tuple([0] * 9)] == 0
