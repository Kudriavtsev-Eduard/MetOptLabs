from functions import Function


class Report:
    def __init__(self, func: Function, tracking: tuple[tuple[float, ...], ...], aborted: bool,
                 hyperparameters: dict[str, float]) -> None:
        pass

    def display(self) -> None:
        pass

    def get_result(self) -> float:
        pass

    def get_raw_tracking(self) -> tuple[tuple[float, ...], ...]:
        pass
