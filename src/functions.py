import random
from typing import Callable
import src.utilities as utilities

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
        self.tracking = False
        self.times_used = 0
        self.negate_multiplier = 1

    def apply(self, *args: float) -> float:
        if self.tracking:
            self.times_used += 1
        assert self.get_arg_count() == len(args)
        return self.function(*args) * self.negate_multiplier

    def get_arg_count(self) -> int:
        return self.function.__code__.co_argcount

    def start_tracking(self) -> None:
        self.tracking = True

    def stop_tracking(self) -> None:
        self.tracking = False

    def get_call_data(self) -> dict[str, int]:
        return {"to_function": self.times_used}

    def negate(self) -> None:
        self.negate_multiplier *= -1


class DerivableFunction(Function):
    def __init__(self, function: Callable[..., float], gradient: tuple[Callable[..., float], ...]):
        super().__init__(function)
        self.gradient = gradient
        self.times_gradient_used = False

    def get_gradient_at(self, *args: float) -> tuple[float, ...]:
        if self.tracking:
            self.times_gradient_used += 1
        return tuple(dF(*args) * self.negate_multiplier for dF in self.gradient)

    def get_func_cross_section(self, current_argument: tuple[float, ...]) -> Callable[[float], float]:
        antigravity = utilities.multiply(self.get_gradient_at(*current_argument), -1)

        def evaluteF1D(t):
            return self.apply(*(utilities.element_wise_addition(current_argument, antigravity, t)))

        return evaluteF1D

    def get_call_data(self) -> dict[str, int]:
        result = super().get_call_data()
        result["to_gradient"] = self.times_gradient_used
        return result


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
        self.__arg_count = function.get_arg_count()

    def get_arg_count(self) -> int:
        return self.__arg_count


class CachedFunction(Function):
    def __init__(self, function: Callable[..., float]):
        super().__init__(function)
        self.cache: dict[tuple[float, ...], float] = dict()

    def apply(self, *args: float) -> float:
        if args in self.cache:
            return self.cache[args]
        result = super().apply(*args)
        self.cache[args] = result
        return result


class NoiseFunction(CachedFunction):
    def __init__(self, function: Callable[..., float], creativity: int = 20):
        assert creativity > 0
        super().__init__(lambda *args: function(**args) + (random.randint(-creativity, creativity) + random.random()))
