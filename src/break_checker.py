from abc import ABC

from src.functions import DerivableFunction


class BreakChecker(ABC):
    def is_done(self, consideration_arguments: list[tuple[float, ...]], iteration_number: int,
                func: DerivableFunction) -> bool:
        pass
