# Task: balanced_parens

Implement the function `is_balanced(s)` in `solution.py`.

## Spec
- Input: a string `s` containing any characters.
- Output: `True` if every `(`, `[`, and `{` in `s` has a matching closing bracket of the same type in the correct order. `False` otherwise.
- Brackets of different types must be properly nested: `([])` is valid, `([)]` is not.
- Non-bracket characters are ignored.
- The empty string returns `True`.

## Example
```
is_balanced("(a + b)")   == True
is_balanced("([])")      == True
is_balanced("([)]")      == False
is_balanced("(((")       == False
is_balanced("")          == True
```

A starter file `solution.py` is provided. Public tests are in `public_tests.py` — run them with `python -m pytest public_tests.py`.
