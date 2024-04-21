import numpy as np

from core.game import Actions, NPuzzle


def test_create_puzzle_with_init_state():
    init_state = [1, 4, 6, 7, 5, 3, 0, 2, 8]
    puzzle = NPuzzle(8, init_state)
    assert puzzle.side == 3
    assert len(puzzle.field) == 3
    assert (puzzle.field == np.array(init_state).reshape((3, 3))).all()
    assert puzzle.move(Actions.SPACE_LEFT) == False
    allowed = puzzle.move(Actions.SPACE_RIGHT)
    assert allowed
    assert (
        puzzle.field == np.array([1, 4, 6, 7, 5, 3, 2, 0, 8]).reshape((puzzle.side, -1))
    ).all()
    allowed = puzzle.move(Actions.SPACE_UP)
    assert allowed
    assert (
        puzzle.field == np.array([1, 4, 6, 7, 0, 3, 2, 5, 8]).reshape((puzzle.side, -1))
    ).all()
    allowed = puzzle.move(Actions.SPACE_LEFT)
    assert allowed
    assert (
        puzzle.field == np.array([1, 4, 6, 0, 7, 3, 2, 5, 8]).reshape((puzzle.side, -1))
    ).all()
    allowed = puzzle.move(Actions.SPACE_DOWN)
    assert allowed
    assert (
        puzzle.field == np.array([1, 4, 6, 2, 7, 3, 0, 5, 8]).reshape((puzzle.side, -1))
    ).all()


def test_repr():
    puzzle = NPuzzle(8, [1, 4, 6, 7, 5, 3, 0, 2, 8])
    assert repr(puzzle) == "Puzzle(8, [1, 4, 6, 7, 5, 3, 0, 2, 8])"


def test_str():
    puzzle = NPuzzle(8, [1, 4, 6, 7, 5, 3, 0, 2, 8])
    assert str(puzzle) == "1 4 6\n7 5 3\n  2 8\n"


def test_goal_state():
    puzzle = NPuzzle(8, list(range(1, 9)) + [0])
    assert puzzle.is_goal_state()


def test_starting_configurations():
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

    init_state = list(range(1, 9)) + [0]
    puzzle = NPuzzle(8, init_state)
    for lst in puzzle._starting_configurations(init_state):
        assert lst == init_state
