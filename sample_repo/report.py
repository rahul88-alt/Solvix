"""Builds a short text report from calculator results, used to exercise
cross-file import relationships in retrieval tests.
"""

from calculator import add, subtract
from utils.strings import truncate


def format_summary(a, b):
    total = add(a, b)
    diff = subtract(a, b)
    return truncate(f"Total: {total}, Difference: {diff}")
