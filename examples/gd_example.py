from src.break_checker import BreakChecker, ArgumentAbsoluteBreakChecker
from src.functions import DerivableFunction, AutomatedDerivableFunction, Function
from src.gradient_optimizer import GradientOptimizer
from src.report import Report
from src.scheduler import DichotomyScheduler, Scheduler


def main():
    scheduler: Scheduler = DichotomyScheduler(1, 128)  # Set learning rate strategy
    break_checker: BreakChecker = ArgumentAbsoluteBreakChecker(10 ** -8)  # Set breakpoint condition checking
    optimizer: GradientOptimizer = GradientOptimizer(scheduler, break_checker,
                                                     limit=10 ** 5)  # Combine all above into gradient optimizer

    func: DerivableFunction = AutomatedDerivableFunction(
        Function(lambda x, y: x ** 2 + y ** 2))  # Create function to optimize
    start_point: tuple[float, float] = (10, 10) # Create starting point for algorithm

    report: Report = optimizer.optimize(func, start_point)  # Create report about optimization process

    report.display()  # Display the report in form of a 3D graph


if __name__ == '__main__':
    main()
