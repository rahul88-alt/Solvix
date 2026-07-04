import unittest
from utils.strings import capitalize_words

class TestStrings(unittest.TestCase):
    def test_capitalize_words(self):
        self.assertEqual(capitalize_words("hello world"), "Hello World")
        self.assertEqual(capitalize_words("python is great"), "Python Is Great")
        self.assertEqual(capitalize_words(""), "")
        self.assertEqual(capitalize_words("singleword"), "Singleword")
