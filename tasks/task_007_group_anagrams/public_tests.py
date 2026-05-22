from solution import group_anagrams


def test_basic():
    assert group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]) == [
        ["bat"], ["eat", "tea", "ate"], ["tan", "nat"],
    ]


def test_empty_input():
    assert group_anagrams([]) == []


def test_single():
    assert group_anagrams(["a"]) == [["a"]]
