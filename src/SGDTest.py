import sgd_optimizer
import functions
import pandas
import scheduler
import break_checker
from ucimlrepo import fetch_ucirepo

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

optimizer = sgd_optimizer.StochasticGradientOptimizer(
    scheduler.GolderRatioScheduler(5, 100),
    break_checker.ArgumentAbsoluteBreakChecker(0.0001),
    hyperfunc,
    45)

report = optimizer.optimize(
    dataset[:100],
    (0., 0., 0., 0., 0., 0.),
    20,
    functions.Elastic(6, 14)
)

ans = report.get_raw_tracking()[-1]

### comparison in console
# for pair in dataset:
#     obj = pair[0]
#     y = pair[1]
#     print("expected:", y, "actual:", predfunc(obj, ans[0], ans[1], ans[2], ans[3], ans[4], ans[5]))

report.display_dataset_comparison(dataset, predfunc)
