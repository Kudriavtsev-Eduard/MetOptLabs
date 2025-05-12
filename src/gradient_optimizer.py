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

    @staticmethod
    def __minmax(current: tuple[float, ...], lower_bound: tuple[float, ...] | None = None,
                 upper_bound: tuple[float, ...] | None = None) -> tuple[float, ...]:
        resulting = current
        if lower_bound is not None:
            resulting = tuple(max(lb, cr) for lb, cr in zip(lower_bound, resulting))
        if upper_bound is not None:
            resulting = tuple(min(cr, ub) for cr, ub in zip(resulting, upper_bound))
        return resulting

    def optimize(self, func: DerivableFunction, starting_point: tuple[float, ...] | None = None,
                 lower_bound: tuple[float, ...] | None = None,
                 upper_bound: tuple[float, ...] | None = None,
                 maximum: bool = False) -> Report:
        assert starting_point is None or len(starting_point) == func.get_arg_count()
        if maximum:
            func.negate()
        func.start_tracking()

        multiplier = -1

        if starting_point is None:
            current_point: tuple[float, ...] = tuple([0] * func.get_arg_count())
        else:
            current_point: tuple[float, ...] = starting_point
        tracking: list[tuple[float, ...]] = [current_point]

        it = 0

        while (not self.__break_checker.is_done(tracking, func)) and it < self.__limit:
            current_point = GradientOptimizer.__minmax(
                (
                    utilities.element_wise_addition(
                        current_point, func.get_gradient_at(*current_point),
                        multiplier * self.__scheduler.get_step_value(current_point, it, func)
                    )
                ),
                lower_bound,
                upper_bound
            )
            tracking.append(current_point)
            it += 1
        func.stop_tracking()
        if maximum:
            func.negate()
        return Report(func, tracking, it == self.__limit,
                      self.__scheduler.get_hyper_parameters(), self.__scheduler.get_name())
