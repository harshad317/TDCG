from solution import merge_intervals


def test_basic_overlap():
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]


def test_touching():
    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]


def test_empty():
    assert merge_intervals([]) == []


def test_unsorted():
    assert merge_intervals([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]


def test_single():
    assert merge_intervals([[3, 7]]) == [[3, 7]]


def test_fully_contained():
    assert merge_intervals([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]


def test_all_overlap_into_one():
    assert merge_intervals([[1, 4], [2, 5], [3, 6]]) == [[1, 6]]


def test_no_overlap():
    assert merge_intervals([[1, 2], [3, 4], [5, 6]]) == [[1, 2], [3, 4], [5, 6]]


def test_does_not_mutate_input():
    src = [[3, 5], [1, 2]]
    snapshot = [list(p) for p in src]
    merge_intervals(src)
    assert src == snapshot


def test_duplicates():
    assert merge_intervals([[1, 3], [1, 3], [1, 3]]) == [[1, 3]]


def test_point_intervals():
    assert merge_intervals([[2, 2], [2, 2], [3, 3]]) == [[2, 2], [3, 3]]
