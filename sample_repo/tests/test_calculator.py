from calculator import Calculator, add, subtract


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 3) == 2


def test_subtract_negative_result():
    assert subtract(3, 5) == -2


def test_calculator_add_tracks_running_total():
    calc = Calculator(start=10)
    assert calc.add(5) == 15


def test_calculator_subtract_tracks_running_total():
    calc = Calculator(start=10)
    assert calc.subtract(4) == 6
