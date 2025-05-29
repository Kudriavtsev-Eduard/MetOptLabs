from functools import wraps
from typing import Any, Callable
from ucimlrepo import fetch_ucirepo
from torch.utils.data import TensorDataset, DataLoader

import torch
import pandas
from src import functions, sgd_optimizer, scheduler, break_checker, report

### dataset
student_performance = fetch_ucirepo(id=320)
feature_cols = ['studytime', 'absences', 'failures']
marks_to_feature_cols = ['G1', 'G2']
mark_col = 'G3'

# features: studytime  absences  failures  G1  G2
x = pandas.concat([student_performance.data.features[feature_cols],
                   student_performance.data.targets[marks_to_feature_cols]],
                  axis=1)
y = student_performance.data.targets[mark_col]

# dataset: ((studytime,  absences,  failures,  G1,  G2), G3)
dataset = [(tuple(map(float, data)), float(yi)) for data, yi in zip(x.values, y.values)]

### test
# L(xi, yi, w) = (yi - w0 - w1x1 - ... - wnxn)

predfunc = lambda obj, w0, w1, w2, w3, w4, w5: w0 + w1 * obj[0] + w2 * obj[1] + w3 * obj[2] + w4 * obj[3] + w5 * obj[4]
hyperfunc = functions.HyperFunction(lambda obj, mark, w0, w1, w2, w3, w4, w5:
                                    (mark - predfunc(obj, w0, w1, w2, w3, w4, w5)) ** 2)

dataset_x = [tuple(map(float, data)) for data in x.values]


def loss_counter(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        loss = fn(*args, **kwargs)
        wrapper.loss_calls += 1
        loss.register_hook(lambda grad: setattr(wrapper, "grad_calls", wrapper.grad_calls + 1))
        return loss
    wrapper.loss_calls = 0
    wrapper.grad_calls = 0
    return wrapper


def count_mean(dataset: list[tuple[tuple[float, ...], float]], w) -> float:
    mean = 0
    for obj, expected in dataset:
        mean += abs(expected - predfunc(obj, *w))
    return mean / len(dataset)


def lib_decay(
        model: torch.nn.Module,
        dataset: list[tuple[tuple[float, ...], float]],
        optimizer_type: type[torch.optim.Optimizer],
        hyperparams: dict[str, Any],
        other_params_to_see: set[str],
        loss_func: Callable,
        name: str,
        epochs: int = 41,
        batch_size: int = 32) -> report.Report:
    inputs, targets = zip(*dataset)
    dataset_x = torch.tensor(inputs, dtype=torch.float32)
    dataset_y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(dataset_x, dataset_y), batch_size=batch_size, shuffle=True)

    model.weight.data.zero_()
    model.bias.data.zero_()

    optim = optimizer_type(model.parameters(), **hyperparams)
    group = optim.param_groups[0]
    hparams = {k: v for k, v in group.items() if k in hyperparams or k in other_params_to_see}
    for k, v in group.items():
        if isinstance(v, tuple):
            hparams.pop(k, None)
            for i, vi in enumerate(v, 1):
                hparams[f"{k}{i}"] = vi

    rep = report.Report(functions.Function(lambda x: 5), [], False, hparams, name)

    for ep in range(epochs):
        for x_batch, y_batch in loader:
            loss = loss_func(model(x_batch), y_batch)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            optim.step()

        w0 = model.bias.data.item()
        w_rest = tuple(model.weight.data.view(-1).tolist())
        rep._tracking.append((w0, *w_rest))

    rep._func_calls = loss_func.loss_calls
    loss_func.loss_calls = 0
    loss_func.grad_calls = 0
    return rep


batch, dim_in, dim_out = 45, 5, 1
model = torch.nn.Linear(dim_in, dim_out)

@loss_counter
def counted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.MSELoss(reduction="sum")(pred, target)

our_optimizer = sgd_optimizer.StochasticGradientOptimizer(
    scheduler.GolderRatioScheduler(0.2, 20),
    break_checker.ArgumentAbsoluteBreakChecker(0.001),
    hyperfunc,
    40
)
reports = [lib_decay(model, dataset, torch.optim.SGD, {"lr": 0.01, "momentum": 0.9, "nesterov": True}, set(), counted_mse, "Nesterov"),
           lib_decay(model, dataset, torch.optim.SGD, {"lr": 0.01, "momentum": 0.85}, set(),counted_mse, "Momentum"),
           lib_decay(model, dataset, torch.optim.Adagrad, {"lr": 0.3}, set(), counted_mse, "Adagrad"),
           lib_decay(model, dataset, torch.optim.RMSprop, {"lr": 0.01, "alpha": 0.9}, set(), counted_mse,"RMSProp"),
           lib_decay(model, dataset, torch.optim.Adam, {"lr": 0.1}, {"betas", "eps"}, counted_mse, "Adam"),
           our_optimizer.optimize(dataset,(0., 0., 0., 0., 0., 0.),45,functions.L2(6, 0))]

for report in reports:
    report._mean_error_value = count_mean(dataset, report.get_raw_tracking()[-1])
    report.display_dataset_comparison(dataset, predfunc)

# w = (w0, *w_rest)
# for obj, mark in dataset:
#     y_pred = predfunc(obj, *w)
#     print(f"expected={mark:.3f}, predicted={y_pred:.3f}, error={(mark-y_pred)**2:.3f}")


### comparison in console
# for pair in dataset:
#     obj = pair[0]
#     y = pair[1]
#     print("expected:", y, "actual:", predfunc(obj, ans[0], ans[1], ans[2], ans[3], ans[4], ans[5]))

# report.display_dataset_comparison(dataset, predfunc)
