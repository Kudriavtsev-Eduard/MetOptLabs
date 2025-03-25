from abc import ABC, abstractmethod
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

    def get_name(self) -> str:
        return self.__class__.__name__


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


class SegmentScheduler(Scheduler, ABC):
    def __init__(self, indent: float, count_iterations: int) -> None:
        self.indent = indent
        self.count_iterations = count_iterations

    def get_step_value(self, current_argument: tuple[float, ...], iteration_number: int,
                       func: DerivableFunction) -> float:
        arg1, arg2 = self.indent, -self.indent
        return self._min_per_segment(func.get_func_cross_section(current_argument), arg1, arg2)

    @abstractmethod
    def _min_per_segment(self, func: Callable[[float], float], a: float, b: float) -> float:
        pass


class DichotomyScheduler(SegmentScheduler):
    @staticmethod
    def __get_middle(a: float, b: float) -> float:
        return a + (b - a) / 2

    def _min_per_segment(self, func: Callable[[float], float], a: float, b: float) -> float:
        n = self.count_iterations
        for i in range(n):
            mid = DichotomyScheduler.__get_middle(a, b)
            left_mid = DichotomyScheduler.__get_middle(a, mid)

            val_m = abs(func(mid))
            val_lm = abs(func(left_mid))
            if val_lm < val_m:
                b = mid
                continue

            right_mid = DichotomyScheduler.__get_middle(mid, b)
            val_rm = abs(func(right_mid))
            if val_rm < val_m:
                a = mid
                continue
            a = left_mid
            b = right_mid

        return DichotomyScheduler.__get_middle(a, b)


class GolderRatioScheduler(SegmentScheduler):
    __left_indent = 0.382
    __right_indent = 1 - __left_indent

    def _min_per_segment(self, func: Callable[[float], float], a: float, b: float) -> float:
        n = self.count_iterations
        delta = b - a
        c = a + GolderRatioScheduler.__left_indent * delta
        d = a + GolderRatioScheduler.__right_indent * delta
        val_c = abs(func(c))
        val_d = abs(func(d))
        for i in range(n):
            if val_c <= val_d:
                b = d
                d = c
                c = a + GolderRatioScheduler.__left_indent * (b - a)
                val_d, val_c = val_c, abs(func(c))
                continue

            a = c
            c = d
            d = a + GolderRatioScheduler.__right_indent * (b - a)
            val_c, val_d = val_d, abs(func(d))

        return c if val_c <= val_d else d
