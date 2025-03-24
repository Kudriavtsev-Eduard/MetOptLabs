

def element_wise_addition(first: tuple[float, ...], second: tuple[float, ...], multiplier: float) -> (
        tuple)[float, ...]:
    assert len(first) == len(second)
    return tuple(a + multiplier * b for a, b in zip(first, second))


def add_point(p1: tuple[float, ...], p2: tuple[float, ...]) -> tuple[float, ...]:
    assert len(p1) == len(p2)
    ans = []
    for i in range(len(p1)):
        ans.append(p1[i] + p2[i])
    return tuple(ans)


def multiply(p: tuple[float, ...], scalar) -> tuple[float, ...]:
    return tuple(scalar * a for a in p)


