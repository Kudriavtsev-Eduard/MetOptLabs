from src.functions import DerivableFunction
from src.scheduler import DichotomyScheduler, GolderRatioScheduler
import scheduler
import utilities

"""
    Testing function for scheduler
"""

def test(func: DerivableFunction, arg: tuple[float, ...], scheduler: scheduler.Scheduler):
    print("test of scheduler:", scheduler.get_name())
    print("Start:", arg, "Value:", func.apply(*arg))
    h = scheduler.get_step_value(arg, 0, func)
    antigravity = utilities.multiply(func.get_gradient_at(*arg), -1)
    print("Antigravity:", antigravity, "h:", h)
    next_arg = utilities.add_point(arg, utilities.multiply(antigravity, h))
    print("End:", next_arg, "Value:", func.apply(*next_arg))
    print("============================")


dihotomy_scheduler = DichotomyScheduler(10, 20)

test(DerivableFunction(lambda x, y: (x - 8) ** 2 + 100 * y ** 2, (lambda x, y: 2 * (x - 8), lambda x, y: 200 * y)),
     (5, 0), dihotomy_scheduler)

test(DerivableFunction(lambda x, y: (x - 8) ** 2 + 100 * y ** 2, (lambda x, y: 2 * (x - 8), lambda x, y: 200 * y)),
     (2, 7), dihotomy_scheduler)

test(DerivableFunction(lambda x, y: 0.01 * x ** 2 + 10 * y ** 2, (lambda x, y: 0.02 * x, lambda x, y: 20 * y)),
     (11, -12), dihotomy_scheduler)



golden_scheduler = GolderRatioScheduler(10, 20)

test(DerivableFunction(lambda x, y: (x - 8) ** 2 + 100 * y ** 2, (lambda x, y: 2 * (x - 8), lambda x, y: 200 * y)),
     (5, 0), golden_scheduler)

test(DerivableFunction(lambda x, y: (x - 8) ** 2 + 100 * y ** 2, (lambda x, y: 2 * (x - 8), lambda x, y: 200 * y)),
     (2, 7), golden_scheduler)

test(DerivableFunction(lambda x, y: 0.01 * x ** 2 + 10 * y ** 2, (lambda x, y: 0.02 * x, lambda x, y: 20 * y)),
     (11, -12), golden_scheduler)


