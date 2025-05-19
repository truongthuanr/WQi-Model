# Temporary 


from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam

def create_sequence_data(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

# Chuẩn hóa
scaler = MinMaxScaler()
alkalinity = df['Độ kiềm'].values.reshape(-1, 1)
alkalinity_scaled = scaler.fit_transform(alkalinity)

# Tạo dữ liệu chuỗi
window_size = 10
X, y = create_sequence_data(alkalinity_scaled, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))  # LSTM expects 3D shape

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Mô hình LSTM
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(optimizer=Adam(0.001), loss='mse')

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

# Dự đoán
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Đánh giá
print("MAE:", mean_absolute_error(y_test_inv, y_pred))
print("R²:", r2_score(y_test_inv, y_pred))

# Plot
plt.figure(figsize=(10,5))
plt.plot(y_test_inv, label="True")
plt.plot(y_pred, label="Predicted", linestyle="dashed")
plt.legend()
plt.title("LSTM - Time Series Forecast")
plt.show()
