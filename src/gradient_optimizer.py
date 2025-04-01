from src.break_checker import BreakChecker
from src.functions import DerivableFunction
from src.report import Report
from src.scheduler import Scheduler
import utilities


class GradientOptimizer:
    def __init__(self, scheduler: Scheduler, break_checker: BreakChecker, limit: int):
        self.__scheduler = scheduler
        self.__break_checker = break_checker
        self.__limit = limit

    def optimize(self, func: DerivableFunction, starting_point: tuple[float, ...] | None = None,
                 maximum: bool = False) -> Report:
        assert starting_point is None or len(starting_point) == func.get_arg_count()

        multiplier = 1 if maximum else -1

        if starting_point is None:
            current_point: tuple[float, ...] = tuple([0] * func.get_arg_count())
        else:
            current_point: tuple[float, ...] = starting_point
        tracking: list[tuple[float, ...]] = [current_point]

        it = 0

        while (not self.__break_checker.is_done(tracking, func)) and it < self.__limit:
            current_point = (
                utilities.element_wise_addition(
                    current_point, func.get_gradient_at(*current_point),
                    multiplier * self.__scheduler.get_step_value(current_point, it, func)
                )
            )
            tracking.append(current_point)
            it += 1

        return Report(func, tracking, it == self.__limit,
                      self.__scheduler.get_hyper_parameters(), self.__scheduler.get_name())
