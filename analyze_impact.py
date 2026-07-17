import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv('final-data.csv')
df['Week Endings'] = pd.to_datetime(df['Week Endings'])
df = df.sort_values('Week Endings').reset_index(drop=True)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

model_2017 = keras.models.load_model('model_2017.keras')
model_2024 = keras.models.load_model('model_2024.keras')

look_back = 12

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

price_data = df['Avg Ticket Price ($)'].values.astype('float32').reshape(-1, 1)
price_scaled = scaler.transform(price_data)

def evaluate_impact(model, start_date, end_date, title):
    mask = (df['Week Endings'] >= pd.to_datetime(start_date)) & (df['Week Endings'] <= pd.to_datetime(end_date))
    eval_df = df[mask].reset_index(drop=True)
    
    if len(eval_df) == 0:
        print(f"No data for {title}")
        return
        
    start_idx = df.index[mask][0]
    
    input_data = price_scaled[start_idx - look_back : start_idx + len(eval_df)]
    
    X, y_true = create_dataset(input_data, look_back=look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    y_pred_scaled = model.predict(X, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y_true.reshape(-1, 1))
    
    avg_actual = np.mean(y_actual)
    avg_pred = np.mean(y_pred)
    pct_diff = ((avg_actual - avg_pred) / avg_pred) * 100
    
    print(f"--- {title} ---")
    print(f"Window: {start_date} to {end_date}")
    print(f"Actual Avg Price: ${avg_actual:.2f}")
    print(f"Predicted Avg Price: ${avg_pred:.2f}")
    print(f"% Difference: {pct_diff:.2f}%\n")
    
    plt.figure(figsize=(10, 5))
    plt.plot(eval_df['Week Endings'], y_actual, label='Actual Price')
    plt.plot(eval_df['Week Endings'], y_pred, label='Predicted (Counterfactual)')
    plt.title(f"{title} - Actual vs Counterfactual")
    plt.legend()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

# Evaluate 2019 Movie Impact (Announcement roughly 2016-09 to 2019-07 Release, let's use 2017-04 to 2019-07)
# Reference project used 18.6% increase. Let's look at 2017-04 to 2019-07
evaluate_impact(model_2017, '2017-04-01', '2019-07-31', '2019 Movie Impact')

# Evaluate COVID stabilization (2020-03 to 2022-03)
evaluate_impact(model_2017, '2021-08-01', '2022-12-31', 'COVID Post-Opening Impact')

# Evaluate 2024 Movie Impact (April 2024 trailer to Dec 2024 release)
evaluate_impact(model_2024, '2024-04-01', '2024-12-31', '2024 Movie Impact')
