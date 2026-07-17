import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

df = pd.read_csv('final-data.csv')
df['Week Endings'] = pd.to_datetime(df['Week Endings'])
df = df.sort_values('Week Endings').reset_index(drop=True)

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

price_data = df['Avg Ticket Price ($)'].values.astype('float32').reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
price_scaled = scaler.fit_transform(price_data)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

look_back = 12 # 12 weeks seems reasonable

print("Training 2017 model...")
cutoff_2017 = pd.to_datetime('2017-04-01')
train_2017 = price_scaled[df['Week Endings'] < cutoff_2017]
X_train1, y_train1 = create_dataset(train_2017, look_back=look_back)
X_train1 = np.reshape(X_train1, (X_train1.shape[0], X_train1.shape[1], 1))

model1 = keras.Sequential([
    keras.layers.SimpleRNN(32, input_shape=(look_back, 1)),
    keras.layers.Dense(1)
])
model1.compile(loss='mean_squared_error', optimizer='adam')
model1.fit(X_train1, y_train1, epochs=20, batch_size=16, verbose=0)
model1.save('model_2017.keras')
print("Saved model_2017.keras")

print("Training 2024 model...")
cutoff_2024 = pd.to_datetime('2024-04-01')
train_2024 = price_scaled[df['Week Endings'] < cutoff_2024]
X_train2, y_train2 = create_dataset(train_2024, look_back=look_back)
X_train2 = np.reshape(X_train2, (X_train2.shape[0], X_train2.shape[1], 1))

model2 = keras.Sequential([
    keras.layers.SimpleRNN(32, input_shape=(look_back, 1)),
    keras.layers.Dense(1)
])
model2.compile(loss='mean_squared_error', optimizer='adam')
model2.fit(X_train2, y_train2, epochs=20, batch_size=16, verbose=0)
model2.save('model_2024.keras')
print("Saved model_2024.keras")
