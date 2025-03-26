from dataclasses import dataclass
from abc import ABC
from typing import Callable

from src.functions import DerivableFunction
from src.utilities import element_wise_addition, norm


@dataclass
class BreakChecker(ABC):
    __epsilon: float
    __desired_length: int
    __check_value_getter: Callable[[list[tuple[float, ...]], DerivableFunction], float]
    _relativity_function: Callable[[tuple[float, ...], DerivableFunction], float] | None = None

    def is_done(self, consideration_arguments: list[tuple[float, ...]], func: DerivableFunction) -> bool:
        if self.__desired_length == 0 and self._relativity_function is not None:
            self.__desired_length += 1
        if len(consideration_arguments) < self.__desired_length:
            return False
        return self.__check_value_getter(consideration_arguments, func) < self.__epsilon * (
            1 if self._relativity_function is None else self._relativity_function(consideration_arguments[-1], func))


class ArgumentAbsoluteBreakChecker(BreakChecker):
    def __init__(self, epsilon: float) -> None:
        super().__init__(epsilon, 2, lambda args, _: norm(element_wise_addition(args[-1], args[-2], -1)))


class ArgumentRelativeBreakChecker(ArgumentAbsoluteBreakChecker):
    def __init__(self, epsilon: float) -> None:
        super().__init__(epsilon)
        self._relativity_function = lambda x, _: norm(x) + 1


class FunctionAbsoluteBreakChecker(BreakChecker):
    def __init__(self, epsilon: float) -> None:
        super().__init__(epsilon, 2, lambda args, func: abs(func.apply(*args[-1]) - func.apply(*args[-2])))


class FunctionRelativeBreakChecker(FunctionAbsoluteBreakChecker):
    def __init__(self, epsilon: float) -> None:
        super().__init__(epsilon)
        self._relativity_function = lambda x, func: abs(func.apply(x)) + 1


class GradientAbsoluteBreakChecker(BreakChecker):
    def __init__(self, epsilon: float):
        super().__init__(epsilon, 1, lambda args, func: norm(func.get_gradient_at(*args[-1])) ** 2)


class GradientRelativeBreakChecker(GradientAbsoluteBreakChecker):
    def __init__(self, epsilon: float):
        super().__init__(epsilon)
        self._relativity_function = lambda x, func: norm(func.get_gradient_at(*x)) ** 2
