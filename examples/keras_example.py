import pandas
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.optimizers import RMSprop
from keras.optimizers import Adam

from src.functions import DerivableFunction
from src.report import Report

import tensorflow as tf

lr = 1.0
batch_size=32
epochs=80
loss_function_calls = 0

def custom_loss(y_true, y_pred):
    global loss_function_calls
    loss_function_calls += 1
    return tf.reduce_mean(tf.square(y_true - y_pred))


def get_arr_weights(model_weights) -> list:
    w = [item[0] for item in model_weights[0].tolist()]
    w = [model_weights[1].tolist()[0]] + w
    return w


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
    Dense(1, activation='linear')
])
optimizer = Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss=custom_loss, metrics=['mean_squared_error'])
model.build(input_shape=(None, 5))

w = []
w.append(get_arr_weights(model.get_weights()))
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
w.append(get_arr_weights(model.get_weights()))
loss, _ = model.evaluate(x, y)

predfunc = lambda obj, w0, w1, w2, w3, w4, w5: w0 + w1 * obj[0] + w2 * obj[1] + w3 * obj[2] + w4 * obj[3] + w5 * obj[4]
f = DerivableFunction((lambda *agrs: 0), (lambda *agrs: 0))
report = Report(f, w, False, {"lr": lr, "batch_size": batch_size, "epochs": epochs}, "keras-Adam", loss, loss_function_calls)
dataset = [(tuple(map(float, data)), float(yi)) for data, yi in zip(x.values, y.values)]
report.display_dataset_comparison(dataset, predfunc)
