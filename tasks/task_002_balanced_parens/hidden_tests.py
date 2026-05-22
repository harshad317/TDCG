from solution import is_balanced


def test_simple_paren():
    assert is_balanced("(a + b)") is True


def test_nested():
    assert is_balanced("([])") is True


def test_unbalanced_open():
    assert is_balanced("(((") is False


def test_unbalanced_close():
    assert is_balanced(")))") is False


def test_empty():
    assert is_balanced("") is True


def test_mismatched_order():
    assert is_balanced("([)]") is False


def test_deep_nesting():
    assert is_balanced("({[({[]})]})") is True


def test_only_non_brackets():
    assert is_balanced("hello world") is True


def test_close_before_open():
    assert is_balanced(")(") is False


def test_curly_mismatch():
    assert is_balanced("{") is False


def test_mixed_valid():
    assert is_balanced("a(b[c]{d}e)f") is True


def test_mixed_invalid():
    assert is_balanced("a(b[c}d]e)") is False
