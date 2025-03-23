import abc
import math

from src.functions import DerivableFunction


class Scheduler:
    def __init__(self, hyperparameters: dict[str, float]):
        self.hyperparameters = hyperparameters

    @abc.abstractmethod
    def get_step_value(self, current_argument: tuple[float], iteration_number: int, func: DerivableFunction) -> float:
        pass


class ExponentialDecay(Scheduler):
    def __init__(self, step0: int, lamda: int):
        assert lamda > 0
        super().__init__({"step0": step0, "lamda": lamda})

    def get_step_value(self, current_argument: tuple[float], iteration_number: int, func: DerivableFunction) -> float:
        return self.hyperparameters["step0"] * math.exp(-self.hyperparameters["lamda"] * iteration_number)


"""
recommended hyperparameters for PolynomialDecay:
alpha=1/2
beta=1
"""


class PolynomialDecay(Scheduler):
    def __init__(self, alpha: int, beta: int):
        assert alpha > 0 and beta > 0
        super().__init__({"alpha": alpha, "beta": beta})

    def get_step_value(self, current_argument: tuple[float], iteration_number: int, func: DerivableFunction) -> float:
        h0 = 1 / math.sqrt(iteration_number + 1)
        return h0 * ((self.hyperparameters["beta"] * iteration_number + 1) ** -self.hyperparameters["alpha"])
