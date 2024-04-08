from core.game import Actions, NPuzzle


def test_create_puzzle_with_init_state():
    puzzle = NPuzzle(8, [1, 4, 6, 7, 5, 3, 0, 2, 8])
    assert puzzle.side == 3
    assert len(puzzle.state) == 9
    assert puzzle.move(Actions.SPACE_LEFT) == False
    puzzle.move(Actions.SPACE_RIGHT)
    assert puzzle.state == [1, 4, 6, 7, 5, 3, 2, 0, 8]
    puzzle.move(Actions.SPACE_UP)
    assert puzzle.state == [1, 4, 6, 7, 0, 3, 2, 5, 8]
    puzzle.move(Actions.SPACE_LEFT)
    assert puzzle.state == [1, 4, 6, 0, 7, 3, 2, 5, 8]
    puzzle.move(Actions.SPACE_DOWN)
    assert puzzle.state == [1, 4, 6, 2, 7, 3, 0, 5, 8]


def test_repr():
    puzzle = NPuzzle(8, [1, 4, 6, 7, 5, 3, 0, 2, 8])
    assert repr(puzzle) == "1 4 6\n7 5 3\n  2 8"
