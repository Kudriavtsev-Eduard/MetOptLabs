from abc import ABC

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
