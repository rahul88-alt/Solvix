"""A tiny calculator module used as a sample repo for indexer tests."""


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def subtract(a, b):
    return a - b


class Calculator:
    """Stateful calculator that remembers a running total."""

    def __init__(self, start=0):
        self.total = start

    def add(self, value):
        self.total = add(self.total, value)
        return self.total

    def subtract(self, value):
        self.total = subtract(self.total, value)
        return self.total
