from solution import roman_to_int


def test_one():
    assert roman_to_int("I") == 1


def test_three():
    assert roman_to_int("III") == 3


def test_four():
    assert roman_to_int("IV") == 4


def test_five():
    assert roman_to_int("V") == 5


def test_nine():
    assert roman_to_int("IX") == 9


def test_fifty_eight():
    assert roman_to_int("LVIII") == 58


def test_forty():
    assert roman_to_int("XL") == 40


def test_ninety():
    assert roman_to_int("XC") == 90


def test_four_hundred():
    assert roman_to_int("CD") == 400


def test_nine_hundred():
    assert roman_to_int("CM") == 900


def test_1994():
    assert roman_to_int("MCMXCIV") == 1994


def test_3999():
    assert roman_to_int("MMMCMXCIX") == 3999


def test_thousand():
    assert roman_to_int("M") == 1000


def test_two_thousand_four():
    assert roman_to_int("MMIV") == 2004


def test_mixed_subtractive():
    assert roman_to_int("XLIX") == 49
