# Task: roman_to_int

Implement the function `roman_to_int(s)` in `solution.py`.

## Spec
- Input: a non-empty string `s` containing a valid Roman numeral using the symbols `I, V, X, L, C, D, M`.
- Output: the integer value as an `int`.
- The standard subtractive forms apply: `IV=4, IX=9, XL=40, XC=90, CD=400, CM=900`. All other adjacent pairs add.
- Assume input is well-formed (no validation required).
- Range covered: 1 through 3999.

## Symbol values
```
I=1, V=5, X=10, L=50, C=100, D=500, M=1000
```

## Example
```
roman_to_int("III")     == 3
roman_to_int("IV")      == 4
roman_to_int("IX")      == 9
roman_to_int("LVIII")   == 58
roman_to_int("MCMXCIV") == 1994
```

A starter file `solution.py` is provided. Public tests are in `public_tests.py` — run them with `python -m pytest public_tests.py`.
