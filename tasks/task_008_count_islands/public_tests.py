from solution import count_islands


def test_empty():
    assert count_islands([]) == 0


def test_single_island():
    assert count_islands([[1]]) == 1


def test_two_islands():
    assert count_islands([[1, 1, 0], [1, 0, 0], [0, 0, 1]]) == 2
