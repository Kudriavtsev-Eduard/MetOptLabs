import typing

from src.break_checker import BreakChecker
from src.gradient_optimizer import GradientOptimizer
from src.report import Report
from src.scheduler import Scheduler
from src.functions import BatchAutomatedDerivableFunction, HyperFunction, DerivableFunction


class StochasticGradientOptimizer:
    def __init__(self, scheduler: Scheduler, break_checker: BreakChecker, hyper_func: HyperFunction, limit: int):
        self.grad_optimizer = GradientOptimizer(scheduler, break_checker, limit)
        self.hyper_func = hyper_func

    def optimize(self, dataset: typing.Sequence[tuple[tuple[float, ...], float]], hyperparams_begin: tuple[float, ...],
                 batch_size: int, regular_func: DerivableFunction):
        to_optimize = BatchAutomatedDerivableFunction(self.hyper_func, dataset, batch_size, regular_func)
        r = self.grad_optimizer.optimize(to_optimize, hyperparams_begin)
        return r, to_optimize.times_used
