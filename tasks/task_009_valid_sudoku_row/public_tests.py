from solution import is_valid_sudoku


def _empty():
    return [["." for _ in range(9)] for _ in range(9)]


def test_empty_board_valid():
    assert is_valid_sudoku(_empty()) is True


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
