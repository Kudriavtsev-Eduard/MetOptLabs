import inspect
from typing import Callable

"""
functions.py
Standard implementation of function classes.

Example of usage:
    my_function = DerivableFunction(lambda x, y: x ** 2 + y ** 2, (lambda x, y: 2 * x, lambda x, y: 2 * y))
    my_function.apply(1, 2) -> 5
    my_function.get_gradient_at(1, 2) -> (2, 4)
"""


class Function:

    def __init__(self, function: Callable[..., float]):
        self.function = function

    def apply(self, *args: float) -> float:
        return self.function(*args)

    def get_arg_count(self) -> int:
        return 0


class DerivableFunction(Function):

    def __init__(self, function: Callable[..., float], gradient: tuple[Callable[..., float], ...]):
        super().__init__(function)
        self.gradient = gradient

    def get_gradient_at(self, *args: float) -> tuple[float, ...]:
        return tuple(dF(*args) for dF in self.gradient)
