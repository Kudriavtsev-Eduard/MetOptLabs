from src.break_checker import BreakChecker
from src.functions import DerivableFunction
from src.report import Report
from src.scheduler import Scheduler
import src.utilities as utilities


class GradientOptimizer:
    def __init__(self, scheduler: Scheduler, break_checker: BreakChecker, limit: int):
        self.__scheduler = scheduler
        self.__break_checker = break_checker
        self.__limit = limit

    def optimize(self, func: DerivableFunction, starting_point: tuple[float, ...] | None = None) -> Report:
        func.start_tracking()

        multiplier = -1

        if starting_point is None:
            current_point: tuple[float, ...] = tuple([0] * func.get_arg_count())
        else:
            current_point: tuple[float, ...] = starting_point
        tracking: list[tuple[float, ...]] = [current_point]

        it = 0

        while (not self.__break_checker.is_done(tracking, func)) and it < self.__limit:
            gradient = func.get_gradient_at(*current_point)
            current_point = (
                utilities.element_wise_addition(
                    current_point, gradient,
                    multiplier * self.__scheduler.get_step_value(it, func.get_directional(current_point))
                )
            )
            tracking.append(current_point)
            it += 1
        func.stop_tracking()
        return Report(func, tracking, it == self.__limit,
                      self.__scheduler.get_hyper_parameters(), self.__scheduler.get_name())
