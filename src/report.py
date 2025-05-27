from dataclasses import dataclass, field
from typing import Any, Sequence, Callable

import plotly.graph_objects as go
import json

from plotly.subplots import make_subplots

from src.functions import Function

"""
report.py
Displays job done by GradientOptimizer. Builds a graph of functions.
"""

DEFAULT_CONFIG_PATH = "../config/display_settings.json"


@dataclass
class Report:
    _func: Function
    _tracking: list[tuple[float, ...]]
    _is_aborted: bool
    _hyperparameters: dict[str, float]
    _strategy_name: str
    _config_path: str = DEFAULT_CONFIG_PATH
    _config: dict = field(init=False)

    def __post_init__(self) -> None:
        with open(self._config_path, "r") as config_file:
            self._config = json.load(config_file)

    def display(self) -> None:
        match self._func.get_arg_count():
            case 2:
                self._build_3d_graph()
            case _:
                raise NotImplementedError("Report supports only functions with 2 args.")

    def display_dataset_comparison(self, dataset: Sequence[tuple[tuple[float, ...], float]], predfunc: Callable) -> None:
        assert dataset is not None
        self._build_comparison_graph(dataset, predfunc)

    def _build_comparison_graph(self, dataset: Sequence[tuple[tuple[float, ...], float]], predfunc: Callable) -> None:
        w = self._tracking[-1]
        settings = self._get_settings("error_comparison")
        graph_settings = settings["graph"]
        alignment_settings = settings["alignment"]
        fig = (make_subplots(
            row_heights=alignment_settings["row_heights"],
            vertical_spacing=alignment_settings["vertical_spacing"],
            specs=alignment_settings["specs"],
            subplot_titles=alignment_settings["subplot_titles"],
            rows=2, cols=1)
            .update_layout(
                xaxis_title='INDEX',
                yaxis_title='MARK VALUE',
                autosize=True)
            .add_hline(
                line_dash=graph_settings["zero_line_dash"],
                annotation_text="ZERO",
                y=0,
                row=2, col=1
            )
            .add_trace(
                self._get_dataset_trace(dataset, graph_settings),
                row=2, col=1
            )
            .add_trace(
                self._get_dataset_error_trace(dataset, tuple(w), predfunc, graph_settings),
                row=2, col=1
            )
            .add_trace(
                self._get_table(self._get_settings("table")),
                row=1, col=1
            )
        )
        for ann in fig.layout.annotations:
            ann.update(xref='x domain', x=0, xanchor='left')
        fig.show(renderer="browser")

    def get_raw_tracking(self) -> list[tuple[float, ...]]:
        return self._tracking

    def _format_point(self, point: tuple[float, ...]):
        return "(" + ", ".join(map(lambda flt: self._format_precision(flt), point)) + ")"

    @staticmethod
    def _format_precision(value: float) -> str:
        return "{:.3f}".format(value).rstrip("0").rstrip(".")

    @staticmethod
    def _get_max_column_proportion(lst: list[list[str]], column: int) -> int:
        return max(map(len, map(lambda pair: pair[column], lst)))

    def _get_settings(self, dict_name: str) -> dict[str, Any]:
        if dict_name not in self._config:
            raise KeyError("No parent dict with name " + dict_name)
        return self._config.get(dict_name)

    def _build_3d_graph(self) -> None:
        fig = (
            go.Figure()
            .add_trace(self._get_table(self._get_settings("table")))
            .add_trace(self._get_graph(self._get_settings("graph")))
            .add_trace(self._get_trace(self._get_settings("trace")))
        )
        fig.update_layout(autosize=True)
        fig.show(renderer="browser")

    def _get_graph(self, settings: dict[str, Any]) -> go.Surface:
        var_range = range(*settings["display_range_bounds"])
        var_list = list(var_range)
        z_rangevalues = [[self._func.apply(x, y) for x in var_range] for y in var_range]
        return go.Surface(
            x=var_list,
            y=var_list,
            z=z_rangevalues,
            colorscale=settings["colorscale"],
            contours={
                "z": {
                    "show": True,
                    "start": min(map(min, z_rangevalues)),
                    "end": max(map(max, z_rangevalues)),
                    "size": settings["level_line_indent"]
                }
            },
            opacity=settings["opacity"]
        )

    def _get_table(self, settings: dict[str, Any]) -> go.Table:
        table_values = [
            ["Iterations", f"{len(self._tracking) - 1}"],
            ["Function call data", f"{", ".join(f"{k}={v}" for k, v in self._func.get_call_data().items())}"],
            ["Begin point", self._format_point(self._tracking[0])],
            ["Begin F", self._format_precision(self._func.apply(*self._tracking[0]))],
            ["Min F", self._format_precision(self._func.apply(*self._tracking[-1]))],
            ["Argmin F", self._format_point(self._tracking[-1])],
            ["Strategy", self._strategy_name],
            ["Aborted?", "YES" if self._is_aborted else "NO"],
            ["Hyperparameters", ", ".join(f"{k}={self._format_precision(v)}" for k, v in self._hyperparameters.items())]
        ]

        x_alignment = settings["x_alignment"]
        proportions = [self._get_max_column_proportion(table_values, i) for i in range(2)]
        return go.Table(
            domain=dict(x=[x_alignment, x_alignment + sum(proportions) / 200],
                        y=settings["y_domain"]),
            header=settings["header"],
            cells=dict(values=list(list(col) for col in zip(*table_values)),
                       fill_color=settings["fill_color"],
                       align=settings["align"]),
            columnwidth=proportions)

    def _get_trace(self, settings: dict[str, Any]) -> go.Scatter3d:
        x_values, y_values = zip(*self._tracking)
        tracking_len = len(self._tracking)
        default_marker = settings["default_marker"]
        special_marker = settings["special_marker"]
        marker_settings = settings["marker_params"].copy()
        marker_settings.update({
            "symbol": (
                    [special_marker] +
                    [default_marker] * (tracking_len - 2) +
                    ([special_marker] if tracking_len > 1 else [])
            )
        })
        return go.Scatter3d(
            x=x_values,
            y=y_values,
            z=[self._func.apply(x, y) for x, y in self._tracking],
            mode=settings["scatter_mode"],
            marker=marker_settings,
            line=settings["line_params"]
        )

    def _get_dataset_trace(self, dataset: Sequence[tuple[tuple[float, ...], float]],
                           settings: dict[str, Any]) -> go.Scatter:
        dataset_len = len(dataset)
        return go.Scatter(
            x=list(range(dataset_len)),
            y=[dataset[i][1] for i in range(dataset_len)],
            mode=settings["scatter_mode"],
            line_color=settings["data_line_color"],
            name=settings["data_graph_name"]
        )

    def _get_dataset_error_trace(self, dataset: Sequence[tuple[tuple[float, ...], float]],
                                 w: tuple[float, ...],
                                 predfunc: Callable,
                                 settings: dict[str, Any]) -> go.Scatter:
        return go.Scatter(
            x=list(range(len(dataset))),
            y=[expected - predfunc(obj, *w) for obj, expected in dataset],
            mode=settings["scatter_mode"],
            line_color=settings["error_line_color"],
            name=settings["error_graph_name"]
        )

# report = Report(Function(
#     lambda x, y: 0.1 * x ** 2 + 3 * y ** 2), [(4, 8), (3, 5), (3, 4), (1, 2), (0, 0)], True, {"a": 5.33, "b": 6.33},
#     "Name")
# report.display()
