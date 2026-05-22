from solution import roman_to_int


def test_simple():
    assert roman_to_int("III") == 3


def test_subtractive_iv():
    assert roman_to_int("IV") == 4


def test_complex():
    assert roman_to_int("MCMXCIV") == 1994
