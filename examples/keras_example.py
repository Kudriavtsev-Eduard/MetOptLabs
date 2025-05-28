import pandas
from keras.src.losses import mean_squared_error
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import Callback

from src.functions import Function, DerivableFunction
from src.report import Report


class GradientCalculationCounter(Callback):
    def __init__(self):
        self.gradient_calculations = 0
        self.loss_calculations = 0

    def on_train_batch_end(self, batch, logs=None):
        self.gradient_calculations += batch

    def on_train_batch_begin(self, batch, logs=None):
        self.loss_calculations += batch



student_performance = fetch_ucirepo(id=320)
feature_cols = ['studytime', 'absences', 'failures']
marks_to_feature_cols = ['G1', 'G2']
mark_col = 'G3'

x = pandas.concat([student_performance.data.features[feature_cols],
                   student_performance.data.targets[marks_to_feature_cols]],
                  axis=1).fillna(0)
y = student_performance.data.targets[mark_col].fillna(0)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(1, activation='linear')  # , kernel_regularizer=l2(0.01))
])

optimizer = SGD(learning_rate=0.001, nesterov=True)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

gradient_counter = GradientCalculationCounter()

model.fit(x_train, y_train, epochs=80, batch_size=32, validation_data=(x_val, y_val), callbacks=[gradient_counter])

y_pred = model.predict(x)

mse = mean_squared_error(y, y_pred)

weights = list(model.get_weights())
w = [item[0] for item in weights[0].tolist()]
w = [[weights[1].tolist()[0]] + w]

predfunc = lambda obj, w0, w1, w2, w3, w4, w5: w0 + w1 * obj[0] + w2 * obj[1] + w3 * obj[2] + w4 * obj[3] + w5 * obj[4]

f = DerivableFunction((lambda *agrs: 0), (lambda *agrs: 0))
f.times_used = gradient_counter.loss_calculations
f.times_gradient_used = gradient_counter.gradient_calculations

loss, _ = model.evaluate(x, y)
report = Report(f, w, False, {}, "keras", loss)

dataset = [(tuple(map(float, data)), float(yi)) for data, yi in zip(x.values, y.values)]
report.display_dataset_comparison(dataset, predfunc)

