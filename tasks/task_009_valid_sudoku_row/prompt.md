# Task: valid_sudoku

Implement the function `is_valid_sudoku(board)` in `solution.py`.

## Spec
- Input: a 9x9 list of lists. Each cell is either:
  - an integer from 1 to 9, or
  - the string `"."` representing an empty cell.
- Output: `True` if the board is currently valid; `False` otherwise.
- A board is valid when:
  1. Every row contains no duplicate digits (empty cells are ignored).
  2. Every column contains no duplicate digits.
  3. Each of the nine 3x3 sub-boxes (top-left at rows/cols 0/3/6) contains no duplicate digits.
- The board does NOT need to be solvable. Partially filled boards are fine.
- Do not mutate the input.

## Example (valid)
```
[
  [5,3,".",".",7,".",".",".","."],
  [6,".",".",1,9,5,".",".","."],
  [".",9,8,".",".",".",".",6,"."],
  [8,".",".",".",6,".",".",".",3],
  [4,".",".",8,".",3,".",".",1],
  [7,".",".",".",2,".",".",".",6],
  [".",6,".",".",".",".",2,8,"."],
  [".",".",".",4,1,9,".",".",5],
  [".",".",".",".",8,".",".",7,9]
]
# is_valid_sudoku(board) == True
```

Changing the top-left `5` to an `8` would make the top-left 3x3 box and the leftmost column invalid (duplicate `8`).

A starter file `solution.py` is provided. Public tests are in `public_tests.py` — run them with `python -m pytest public_tests.py`.
