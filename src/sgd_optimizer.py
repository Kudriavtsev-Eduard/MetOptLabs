import typing

from src.break_checker import BreakChecker
from src.gradient_optimizer import GradientOptimizer
from src.report import Report
from src.scheduler import Scheduler
from src.functions import BatchAutomatedDerivableFunction, HyperFunction, DerivableFunction, SumFunction


class StochasticGradientOptimizer:
    def __init__(self, scheduler: Scheduler, break_checker: BreakChecker, hyper_func: HyperFunction, limit: int):
        self.grad_optimizer = GradientOptimizer(scheduler, break_checker, limit)
        self.hyper_func = hyper_func

    def optimize(self, dataset: typing.Sequence[tuple[tuple[float, ...], float]], hyperparams_begin: tuple[float, ...],
                 batch_size: int) -> Report:
        to_optimize = BatchAutomatedDerivableFunction(self.hyper_func, dataset, batch_size)
        return self.grad_optimizer.optimize(to_optimize, hyperparams_begin)

    def optimizeWithReg(self, dataset: typing.Sequence[tuple[tuple[float, ...], float]], hyperparams_begin: tuple[float, ...],
                 batch_size: int, regular_func: DerivableFunction) -> Report:
        to_optimize = BatchAutomatedDerivableFunction(self.hyper_func, dataset, batch_size)
        with_reg = SumFunction(to_optimize, regular_func)
        return self.grad_optimizer.optimize(with_reg, hyperparams_begin)
