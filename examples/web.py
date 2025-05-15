from src.external.terrain_client import TerrainClient
from src.functions import DerivableFunction, AutomatedDerivableFunction, Function, CachedFunction
from src.report import Report
from src.break_checker import BreakChecker, ArgumentAbsoluteBreakChecker, NeverBreakChecker
from src.gradient_optimizer import GradientOptimizer
from src.scheduler import Scheduler, GolderRatioScheduler


def get_default_optimizer():
    scheduler: Scheduler = GolderRatioScheduler(0.005, 10)  # Set learning rate strategy
    break_checker: BreakChecker = ArgumentAbsoluteBreakChecker(0.1)  # Set breakpoint condition checking
    optimizer: GradientOptimizer = GradientOptimizer(scheduler, break_checker,
                                                     limit=100)  # Combine all above into gradient optimizer
    return optimizer


LOWER_BOUND = 40.0
UPPER_BOUND = 43.0
INDENT = 0.1


def get_client() -> TerrainClient | None:
    client = TerrainClient()

    success = client.create_model(
        lat=LOWER_BOUND - INDENT,
        lon=LOWER_BOUND - INDENT,
        lat_end=UPPER_BOUND + INDENT,
        lon_end=UPPER_BOUND + INDENT,
        model_name="Caucasus"
    )

    if not success:
        return None

    return client


def min_max(a: float, mn: float, mx: float) -> float:
    return min(max(a, mn), mx)


def main():
    optimizer: GradientOptimizer = get_default_optimizer()
    client = get_client()
    if client is None:
        return
    func: DerivableFunction = AutomatedDerivableFunction(
        CachedFunction(
            lambda x, y: client.get_elevation(min_max(x, LOWER_BOUND, UPPER_BOUND),
                                              min_max(y, LOWER_BOUND, UPPER_BOUND))), 1)
    start_point: tuple[float, float] = (40.00, 41.30)
    lower_bound = (LOWER_BOUND, LOWER_BOUND)
    upper_bound = (UPPER_BOUND, UPPER_BOUND)

    report: Report = optimizer.optimize(func, start_point, lower_bound, upper_bound, maximum=True)

    print(f"Local peak: {report.get_raw_tracking()[-1]}")


if __name__ == '__main__':
    main()
