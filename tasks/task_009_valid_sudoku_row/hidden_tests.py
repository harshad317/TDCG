from solution import is_valid_sudoku


def _empty():
    return [["." for _ in range(9)] for _ in range(9)]


VALID_BOARD = [
    [5, 3, ".", ".", 7, ".", ".", ".", "."],
    [6, ".", ".", 1, 9, 5, ".", ".", "."],
    [".", 9, 8, ".", ".", ".", ".", 6, "."],
    [8, ".", ".", ".", 6, ".", ".", ".", 3],
    [4, ".", ".", 8, ".", 3, ".", ".", 1],
    [7, ".", ".", ".", 2, ".", ".", ".", 6],
    [".", 6, ".", ".", ".", ".", 2, 8, "."],
    [".", ".", ".", 4, 1, 9, ".", ".", 5],
    [".", ".", ".", ".", 8, ".", ".", 7, 9],
]


def test_empty_valid():
    assert is_valid_sudoku(_empty()) is True


def test_classic_valid():
    assert is_valid_sudoku(VALID_BOARD) is True


def test_row_duplicate():
    b = _empty()
    b[0][0] = 5
    b[0][4] = 5
    assert is_valid_sudoku(b) is False


def test_column_duplicate():
    b = _empty()
    b[0][0] = 5
    b[4][0] = 5
    assert is_valid_sudoku(b) is False


def test_box_duplicate_not_row_or_col():
    # Same 3x3 box, different row and column
    b = _empty()
    b[0][0] = 5
    b[1][1] = 5
    assert is_valid_sudoku(b) is False


def test_corrupt_classic_makes_box_invalid():
    bad = [row[:] for row in VALID_BOARD]
    bad[0][0] = 8  # top-left box already has 8 at (3,0) col-wise; also column 0 conflict
    assert is_valid_sudoku(bad) is False


def test_does_not_mutate():
    import copy
    snapshot = copy.deepcopy(VALID_BOARD)
    is_valid_sudoku(VALID_BOARD)
    assert VALID_BOARD == snapshot


def test_each_digit_once_per_row_ok():
    b = _empty()
    b[0] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert is_valid_sudoku(b) is True


def test_repeats_across_unrelated_rows():
    b = _empty()
    b[0][0] = 1
    b[1][3] = 1  # different row, different col, different box -> ok
    assert is_valid_sudoku(b) is True


def test_box_boundary_no_conflict():
    # cells (2,2) and (3,3) sit in different boxes
    b = _empty()
    b[2][2] = 7
    b[3][3] = 7
    assert is_valid_sudoku(b) is True
