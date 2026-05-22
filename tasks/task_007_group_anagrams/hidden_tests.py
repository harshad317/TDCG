from solution import group_anagrams


def test_basic():
    assert group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]) == [
        ["bat"], ["eat", "tea", "ate"], ["tan", "nat"],
    ]


def test_empty_input():
    assert group_anagrams([]) == []


def test_single_empty_string():
    assert group_anagrams([""]) == [[""]]


def test_single_word():
    assert group_anagrams(["a"]) == [["a"]]


def test_duplicates_preserved():
    assert group_anagrams(["ab", "ba", "ab"]) == [["ab", "ba", "ab"]]


def test_all_unique():
    assert group_anagrams(["foo", "bar", "baz"]) == [["bar"], ["baz"], ["foo"]]


def test_all_same_anagram():
    assert group_anagrams(["abc", "cab", "bca"]) == [["abc", "cab", "bca"]]


def test_groups_sorted_by_key():
    # keys: "abt" ("bat","tab"), "act" ("cat") => "abt" group first
    assert group_anagrams(["cat", "bat", "tab"]) == [["bat", "tab"], ["cat"]]


def test_within_group_input_order():
    assert group_anagrams(["tea", "ate", "eat"]) == [["tea", "ate", "eat"]]


def test_mixed_lengths():
    result = group_anagrams(["a", "b", "ab", "ba"])
    assert result == [["a"], ["ab", "ba"], ["b"]]
