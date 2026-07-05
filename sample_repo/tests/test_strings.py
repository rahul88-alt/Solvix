from utils.strings import count_vowels

def test_count_vowels():
    assert count_vowels("") == 0
    assert count_vowels("abcde") == 2
    assert count_vowels("AEIOU") == 5
    assert count_vowels("bcdfg") == 0
    assert count_vowels("Hello, World!") == 3
