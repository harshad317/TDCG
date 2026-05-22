from solution import caesar_decode


def test_basic_hello():
    assert caesar_decode("Khoor, Zruog!", 3) == "Hello, World!"


def test_zero():
    assert caesar_decode("ABC", 0) == "ABC"


def test_wrap_lower():
    assert caesar_decode("abc", 1) == "zab"


def test_wrap_upper():
    assert caesar_decode("ABC", 1) == "ZAB"


def test_negative_shift():
    assert caesar_decode("xyz", -2) == "zab"


def test_shift_26():
    assert caesar_decode("Hello", 26) == "Hello"


def test_shift_27_equals_1():
    assert caesar_decode("bcd", 27) == "abc"


def test_large_shift():
    assert caesar_decode("abc", 52) == "abc"


def test_preserve_non_letters():
    assert caesar_decode("1234!?", 5) == "1234!?"


def test_mixed_case_preserved():
    assert caesar_decode("AbCdEf", 1) == "ZaBcDe"


def test_empty():
    assert caesar_decode("", 5) == ""


def test_only_spaces():
    assert caesar_decode("   ", 3) == "   "


def test_alphabet_round_trip():
    assert caesar_decode("BCDEFGHIJKLMNOPQRSTUVWXYZA", 1) == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
