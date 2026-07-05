import unittest
from utils.strings import count_vowels

class TestStrings(unittest.TestCase):
    def test_count_vowels(self):
        self.assertEqual(count_vowels("hello"), 2)  # 'e', 'o'
        self.assertEqual(count_vowels("HELLO"), 2)  # 'E', 'O'
        self.assertEqual(count_vowels("Python"), 1) # 'o'
        self.assertEqual(count_vowels(""), 0)       # no vowels
