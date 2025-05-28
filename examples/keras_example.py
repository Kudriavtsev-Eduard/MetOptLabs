import pandas
from keras.src.losses import mean_squared_error
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import SGD


student_performance = fetch_ucirepo(id=320)
feature_cols = ['studytime', 'absences', 'failures']
marks_to_feature_cols = ['G1', 'G2']
mark_col = 'G3'


x = pandas.concat([student_performance.data.features[feature_cols],
                   student_performance.data.targets[marks_to_feature_cols]],
                  axis=1).fillna(0)
y = student_performance.data.targets[mark_col].fillna(0)

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation=LeakyReLU(alpha=0.01), input_dim=x.shape[1]),
    Dense(32, activation=LeakyReLU(alpha=0.01)),
    Dense(1, activation='linear')
])

optimizer = SGD(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

model.fit(x_train, y_train, epochs=50, batch_size=18, validation_data=(x_val, y_val))

# Predict the outputs for the entire dataset
y_pred = model.predict(x)

# Compare the predicted outputs with the original dataset values
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error between actual and predicted values: {mse}")


loss, mse = model.evaluate(x, y)
print(f"Validation Loss: {loss}, Validation MSE: {mse}")
