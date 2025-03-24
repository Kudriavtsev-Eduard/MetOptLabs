

def element_wise_addition(first: tuple[float, ...], second: tuple[float, ...], multiplier: float) -> (
        tuple)[float, ...]:
    assert len(first) == len(second)
    return tuple(a + multiplier * b for a, b in zip(first, second))


def add_point(p1: tuple[float, ...], p2: tuple[float, ...]) -> tuple[float, ...]:
    return element_wise_addition(p1, p2, 1)


def multiply(p: tuple[float, ...], scalar) -> tuple[float, ...]:
    return tuple(scalar * a for a in p)


