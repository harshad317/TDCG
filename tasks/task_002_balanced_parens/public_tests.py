from solution import is_balanced


def test_simple_paren():
    assert is_balanced("(a + b)") is True


def test_nested():
    assert is_balanced("([])") is True


def test_unbalanced():
    assert is_balanced("(((") is False


def test_empty():
    assert is_balanced("") is True
