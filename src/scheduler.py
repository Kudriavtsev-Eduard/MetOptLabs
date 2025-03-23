from abc import ABC

from functions import DerivableFunction


class Scheduler(ABC):
    def __init__(self, hyperparameters: dict[str, float]):
        self.hyperparameters = hyperparameters

    def get_step_value(self, current_argument: tuple[float, ...], iteration_number: int,
                       func: DerivableFunction) -> float:
        pass

    def get_hyper_parameters(self):
        return self.hyperparameters
