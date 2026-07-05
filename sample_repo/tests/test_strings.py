import unittest
from utils.strings import count_vowels

class TestStrings(unittest.TestCase):
    def test_count_vowels(self):
        self.assertEqual(count_vowels("hello"), 2)
        self.assertEqual(count_vowels("HELLO"), 2)
        self.assertEqual(count_vowels("Python"), 1)
        self.assertEqual(count_vowels(""), 0)
        self.assertEqual(count_vowels("bcdfg"), 0)
        self.assertEqual(count_vowels("aeiouAEIOU"), 10)

if __name__ == '__main__':
    unittest.main()
