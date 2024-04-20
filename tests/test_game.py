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
