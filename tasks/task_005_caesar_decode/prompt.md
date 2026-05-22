# Task: caesar_decode

Implement the function `caesar_decode(s, shift)` in `solution.py`.

## Spec
- Input:
  - `s`: a string containing any characters.
  - `shift`: an integer (can be negative, zero, or larger than 26).
- Output: a new string where every ASCII letter has been shifted *backwards* by `shift` positions in the alphabet. Case is preserved. Non-letter characters are passed through unchanged.
- Wrap around: shifting `A` back by 1 gives `Z`. Shifting `a` back by 1 gives `z`.
- A shift of 27 is equivalent to a shift of 1. A shift of -1 is equivalent to a forward shift of 1.

## Examples
```
caesar_decode("Khoor, Zruog!", 3)   == "Hello, World!"
caesar_decode("abc", 1)             == "zab"
caesar_decode("ABC", 0)             == "ABC"
caesar_decode("xyz", -2)            == "zab"
caesar_decode("Hello", 26)          == "Hello"
```

A starter file `solution.py` is provided. Public tests are in `public_tests.py` — run them with `python -m pytest public_tests.py`.
