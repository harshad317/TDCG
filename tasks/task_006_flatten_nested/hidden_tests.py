from solution import flatten


def test_basic():
    assert flatten([1, [2, 3], 4]) == [1, 2, 3, 4]


def test_deep_nesting():
    assert flatten([1, [2, [3, [4]]]]) == [1, 2, 3, 4]


def test_empty():
    assert flatten([]) == []


def test_only_empty_lists():
    assert flatten([[], [], []]) == []


def test_strings_atomic():
    assert flatten(["a", ["b", ["c"]]]) == ["a", "b", "c"]


def test_empty_pockets():
    assert flatten([1, [], [2, []], 3]) == [1, 2, 3]


def test_mixed_types():
    assert flatten([1, ["x", [2, "y"]]]) == [1, "x", 2, "y"]


def test_flat_unchanged():
    assert flatten([1, 2, 3]) == [1, 2, 3]


def test_does_not_mutate():
    src = [1, [2, [3]], 4]
    import copy
    snapshot = copy.deepcopy(src)
    flatten(src)
    assert src == snapshot


def test_deeply_nested_single():
    assert flatten([[[[[[42]]]]]]) == [42]


def test_strings_not_split():
    assert flatten(["abc", ["de"]]) == ["abc", "de"]
