from abc import ABC
from typing import Callable

from functions import DerivableFunction
import math


class Scheduler(ABC):
    __AUXILIARY_PREFIX = "aux_"

    def get_step_value(self, current_argument: tuple[float, ...], iteration_number: int,
                       func: DerivableFunction) -> float:
        return 0

    def get_hyper_parameters(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.__dict__.items() if
                not key.startswith(Scheduler.__AUXILIARY_PREFIX)}


class ExponentialDecayScheduler(Scheduler):
    def __init__(self, step0: float, lamda: float) -> None:
        assert lamda > 0 and step0 > 0
        self.step0 = step0
        self.lamda = lamda

    def get_step_value(self, current_argument: tuple[float, ...], iteration_number: int,
                       func: DerivableFunction) -> float:
        return self.step0 * math.exp(-self.lamda * iteration_number)


class PolynomialDecayScheduler(Scheduler):
    def __init__(self, alpha: float = 1 / 2, beta: float = 1):
        assert alpha > 0 and beta > 0
        self.alpha = alpha
        self.beta = beta

    def get_step_value(self, current_argument: tuple[float, ...], iteration_number: int,
                       func: DerivableFunction) -> float:
        h0 = 1 / math.sqrt(iteration_number + 1)
        return h0 * ((self.beta * iteration_number + 1) ** -self.alpha)


def add_point(p1: tuple[float, ...], p2: tuple[float, ...]) -> tuple[float, ...]:
    ans = []
    for i in range(len(p1)):
        ans.append(p1[i] + p2[i])
    return tuple(ans)


def multiply(p: tuple[float, ...], scalar) -> tuple[float, ...]:
    return tuple(scalar * a for a in p)


class DihotomyScheduler(Scheduler):
    def __init__(self, indent: float, count_iterations: int) -> None:
        self.indent = indent
        self.count_iterations = count_iterations

    def __get_middle(self, a: float, b: float) -> float:
        return a + (b - a) / 2

    def __dihotomy(self, func: Callable[[float], float], a: float, b: float) -> float:
        n = self.count_iterations
        for i in range(n):
            m = self.__get_middle(a, b)
            lm = self.__get_middle(a, m)

            val_m = abs(func(m))
            val_lm = abs(func(lm))
            if val_lm < val_m:
                b = m
                continue

            rm = self.__get_middle(m, b)
            val_rm = abs(func(rm))
            if val_rm < val_m:
                a = m
                continue
            a = lm
            b = rm

        return self.__get_middle(a, b)

    def __get_func_cross_section(self, current_argument: tuple[float, ...], func: DerivableFunction):
        antigravity = multiply(func.get_gradient_at(*current_argument), -1)

        def evaluteF1D(t):
            return func.apply(*(add_point(current_argument, multiply(antigravity, t))))

        return evaluteF1D

    def get_step_value(self, current_argument: tuple[float, ...], iteration_number: int,
                       func: DerivableFunction) -> float:

        func_cross_section = self.__get_func_cross_section(current_argument, func)
        arg1, arg2 = self.indent, -self.indent
        return self.__dihotomy(func_cross_section, arg1, arg2)




