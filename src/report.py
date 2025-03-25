import plotly.graph_objects as go

from functions import Function

"""
report.py
Displays job done by GradientOptimizer. Builds a graph of functions.
"""


class Report:

    def __init__(self, func: Function, display_range: range, tracking: list[tuple[float, ...]],
                 is_aborted: bool, hyperparameters: dict[str, float], strategy_name: str):
        self._func = func
        self._display_range = display_range
        self._tracking = tracking
        self._is_aborted = is_aborted
        self._hyperparameters = hyperparameters
        self._strategy_name = strategy_name

    def get_raw_tracking(self) -> list[tuple[float, ...]]:
        return self._tracking

    def display(self) -> None:
        match self._func.get_arg_count():
            case 2:
                self.build_3d_graph()
            case _:
                raise NotImplementedError("Report supports only functions with 2 args.")

    def build_3d_graph(self) -> None:
        x_values, y_values = zip(*self._tracking)
        z_rangevalues = [[self._func.apply(x, y) for x in self._display_range] for y in self._display_range]
        markers = ['diamond']
        if len(x_values) > 1:
            markers += ['circle'] * (len(x_values) - 2) + ['diamond']

        table_alignment_x = 0.05
        table_values = [
            ["Iterations", f"{len(self._tracking) - 1}"],
            ["Begin point", f"({self._tracking[0][0]}, {self._tracking[0][1]})"],
            ["Begin F", f"{self._func.apply(*self._tracking[0])}"],
            ["Min F", f"{self._func.apply(self._tracking[-1][0], self._tracking[-1][1])}"],
            ["Argmin F", f"({self._tracking[-1][0]}, {self._tracking[-1][1]})"],
            ["Strategy", self._strategy_name],
            ["Aborted?", "YES" if self._is_aborted else "NO"],
            ["Hyperparameters", ", ".join(f"{k}={v}" for k, v in self._hyperparameters.items())]
        ]
        table_header = dict(
            values=["<b>Parameter</b>", "<b>Value</b>"],
            fill_color='lightblue',
            align='center',
        )
        table_cells = dict(
            values=list(list(col) for col in zip(*table_values)),
            fill_color='lavender',
            align='left'
        )

        left_max_proportion = self._get_max_column_proportion(table_values, 0)
        right_max_proportion = self._get_max_column_proportion(table_values, 1)
        fig = (go.Figure()
            .add_trace(go.Surface(
                x=list(self._display_range),
                y=list(self._display_range),
                z=z_rangevalues,
                colorscale='Sunsetdark',
                contours={
                    "z": {
                        "show": True,
                        "start": min(map(min, z_rangevalues)),
                        "end": max(map(max, z_rangevalues)),
                        "size": 5
                    }
                },
                opacity=0.5)
            )
            .add_trace(go.Scatter3d(
                x=x_values,
                y=y_values,
                z=[self._func.apply(x, y) for x, y in self._tracking],
                mode='markers+lines',
                marker=dict(size=7, color='royalblue', symbol=markers),
                line=dict(width=7, color='royalblue'))
            )
            .add_trace(go.Table(
                domain=dict(x=[table_alignment_x, table_alignment_x + (left_max_proportion + right_max_proportion) / 200],
                            y=[0.5, 1]),
                header=table_header,
                cells=table_cells,
                columnwidth=[left_max_proportion, right_max_proportion]))
        )
        fig.update_layout(autosize=True)
        fig.show()

    def _get_max_column_proportion(self, lst: list[list[str]], column: int) -> int:
        return max(map(len, map(lambda pair: pair[column], lst)))

# report = Report(Function(lambda x, y: 0.1 * x ** 2 + 3 * y ** 2), range(-10, 10),[(4, 8), (3, 5) , (3, 4), (1, 2), (0, 0)], True, {"a": 5.33, "b": 6.33}, "Name")
# report.build_3d_graph()
