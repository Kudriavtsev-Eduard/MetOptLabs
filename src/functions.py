import random
from copy import deepcopy
from typing import Callable
import utilities

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
        assert self.get_arg_count() == len(args)
        return self.function(*args)

    def get_arg_count(self) -> int:
        return self.function.__code__.co_argcount


class DerivableFunction(Function):
    def __init__(self, function: Callable[..., float], gradient: tuple[Callable[..., float], ...]):
        super().__init__(function)
        self.gradient = gradient

    def get_gradient_at(self, *args: float) -> tuple[float, ...]:
        return tuple(dF(*args) for dF in self.gradient)

    def get_func_cross_section(self, current_argument: tuple[float, ...]) -> Callable[[float], float]:
        antigravity = utilities.multiply(self.get_gradient_at(*current_argument), -1)

        def evaluteF1D(t):
            return self.apply(*(utilities.element_wise_addition(current_argument, antigravity, t)))

        return evaluteF1D


class AutomatedDerivableFunction(DerivableFunction):

    @staticmethod
    def __get_partial(function: Function, x: tuple[float, ...], coord: int, epsilon: float):
        x_shift = x[:coord] + (x[coord] + epsilon,) + x[coord + 1:]
        return (function.apply(*x_shift) - function.apply(*x)) / epsilon

    def __init__(self, function: Function, epsilon: float = 10 ** -8):
        super().__init__(function.apply,
                         tuple(
                             lambda *x: AutomatedDerivableFunction.__get_partial(function, x, i, epsilon)
                             for i in range(function.get_arg_count()))
                         )


class NoiseFunction(Function):
    def __init__(self, function: Callable[..., float], creativity: float = 1):
        assert creativity > 0
        super().__init__(function)
        self.cache: dict[tuple[float, ...], float] = dict()
        self.creativity = creativity

    def apply(self, *args: float) -> float:
        result = super().apply(*args)
        if args in self.cache:
            offset = self.cache[args]
        else:
            offset = random.random() * (-1 ** random.randint(0, 1)) * self.creativity
            self.cache[args] = offset
        return result + offset
