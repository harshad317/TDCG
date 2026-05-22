from solution import sum_evens


def test_basic():
    assert sum_evens([1, 2, 3, 4]) == 6


def test_empty():
    assert sum_evens([]) == 0


def test_negatives():
    assert sum_evens([-2, -1, 0]) == -2


def test_all_evens():
    assert sum_evens([2, 4, 6, 8]) == 20


def test_single_odd():
    assert sum_evens([7]) == 0


def test_single_even():
    assert sum_evens([8]) == 8


def test_zero_only():
    assert sum_evens([0, 0, 0]) == 0


def test_large():
    assert sum_evens(list(range(1, 101))) == sum(range(2, 101, 2))


def test_mixed_signs():
    assert sum_evens([-4, 3, -2, 1, 6]) == 0
