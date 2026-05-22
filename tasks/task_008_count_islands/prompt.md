# Task: count_islands

Implement the function `count_islands(grid)` in `solution.py`.

## Spec
- Input: a 2D list `grid` of integers, where `1` represents land and `0` represents water.
  - `grid` may be empty (`[]`).
  - Rows may have length 0.
  - All non-empty rows have the same length.
- Output: an integer — the number of islands.
- An island is a maximal group of `1`s connected horizontally or vertically (NOT diagonally).
- Cells outside the grid are treated as water.
- Do not mutate the input.

## Examples
```
count_islands([])                                == 0
count_islands([[0,0,0]])                         == 0
count_islands([[1]])                             == 1
count_islands([[1,1,0],[1,0,0],[0,0,1]])         == 2
count_islands([[1,0,1],[0,1,0],[1,0,1]])         == 5   # diagonals don't connect
count_islands([[1,1,1],[0,1,0],[1,1,1]])         == 1
```

A starter file `solution.py` is provided. Public tests are in `public_tests.py` — run them with `python -m pytest public_tests.py`.
