from solution import sum_evens


def test_basic():
    assert sum_evens([1, 2, 3, 4]) == 6


def test_empty():
    assert sum_evens([]) == 0


def test_no_evens():
    assert sum_evens([1, 3, 5]) == 0
