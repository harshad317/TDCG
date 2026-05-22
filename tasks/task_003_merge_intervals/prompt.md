# Task: merge_intervals

Implement the function `merge_intervals(intervals)` in `solution.py`.

## Spec
- Input: a list of `[start, end]` integer pairs where `start <= end`. The list may be empty or unsorted.
- Output: a new list of `[start, end]` pairs, sorted by start ascending, with all overlapping or touching intervals merged into one.
- Two intervals overlap or touch if `a.end >= b.start` (after sorting by start).
- Do not mutate the input.

## Example
```
merge_intervals([[1,3],[2,6],[8,10],[15,18]])  == [[1,6],[8,10],[15,18]]
merge_intervals([[1,4],[4,5]])                  == [[1,5]]
merge_intervals([])                             == []
merge_intervals([[5,6],[1,2]])                  == [[1,2],[5,6]]
```

A starter file `solution.py` is provided. Public tests are in `public_tests.py` — run them with `python -m pytest public_tests.py`.
