from solution import flatten


def test_basic():
    assert flatten([1, [2, 3], 4]) == [1, 2, 3, 4]


def test_deep():
    assert flatten([1, [2, [3, [4]]]]) == [1, 2, 3, 4]


def test_empty():
    assert flatten([]) == []
