# Task: group_anagrams

Implement the function `group_anagrams(words)` in `solution.py`.

## Spec
- Input: a list of lowercase strings (may be empty, may contain duplicates).
- Output: a list of groups. Each group is a list of strings from the input that are anagrams of each other.
- An anagram means the two strings contain the exact same multiset of characters.
- The empty string is an anagram of itself.

## Ordering rules
- Within each group, strings appear in the same order as they appeared in the input.
- The groups themselves are sorted: by the sorted-letters key of the group, ascending (so the group whose letters sort to `"abt"` comes before the group whose letters sort to `"act"`).

## Examples
```
group_anagrams(["eat","tea","tan","ate","nat","bat"])
  == [["bat"], ["eat","tea","ate"], ["tan","nat"]]

group_anagrams([])         == []
group_anagrams([""])       == [[""]]
group_anagrams(["a"])      == [["a"]]
group_anagrams(["ab","ba","ab"]) == [["ab","ba","ab"]]
```

A starter file `solution.py` is provided. Public tests are in `public_tests.py` — run them with `python -m pytest public_tests.py`.
