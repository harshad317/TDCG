from solution import count_islands


def test_empty():
    assert count_islands([]) == 0


def test_all_water():
    assert count_islands([[0, 0, 0], [0, 0, 0]]) == 0


def test_all_land():
    assert count_islands([[1, 1, 1], [1, 1, 1]]) == 1


def test_single_cell():
    assert count_islands([[1]]) == 1


def test_two_islands():
    assert count_islands([[1, 1, 0], [1, 0, 0], [0, 0, 1]]) == 2


def test_diagonals_do_not_connect():
    assert count_islands([[1, 0, 1], [0, 1, 0], [1, 0, 1]]) == 5


def test_h_shape_one_island():
    assert count_islands([[1, 0, 1], [1, 1, 1], [1, 0, 1]]) == 1


def test_ring_shape():
    assert count_islands([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) == 1


def test_single_row():
    assert count_islands([[1, 0, 1, 1, 0, 1]]) == 3


def test_single_column():
    assert count_islands([[1], [0], [1], [1]]) == 2


def test_does_not_mutate():
    src = [[1, 0], [0, 1]]
    import copy
    snapshot = copy.deepcopy(src)
    count_islands(src)
    assert src == snapshot


def test_larger():
    grid = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1],
    ]
    assert count_islands(grid) == 4
