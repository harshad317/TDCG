from solution import caesar_decode


def test_basic():
    assert caesar_decode("Khoor, Zruog!", 3) == "Hello, World!"


def test_zero_shift():
    assert caesar_decode("ABC", 0) == "ABC"


def test_wrap():
    assert caesar_decode("abc", 1) == "zab"
