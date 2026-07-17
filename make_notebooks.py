import nbformat as nbf

def create_notebook(filename, cells_content):
    nb = nbf.v4.new_notebook()
    cells = []
    for ctype, content in cells_content:
        if ctype == 'md':
            cells.append(nbf.v4.new_markdown_cell(content))
        elif ctype == 'code':
            cells.append(nbf.v4.new_code_cell(content))
    nb['cells'] = cells
    with open(filename, 'w') as f:
        nbf.write(nb, f)

# 1. sequence-model.ipynb
seq_model_cells = [
    ('md', '# Broadway Lion King - Sequence Model for Causal Impact\n\nThis notebook sets up the baseline SimpleRNN forecasting models.'),
    ('code', 'import os\nos.environ["KERAS_BACKEND"] = "torch"\nimport keras\nimport pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import MinMaxScaler\nimport matplotlib.pyplot as plt\nimport pickle'),
    ('code', 'df = pd.read_csv("final-data.csv")\ndf["Week Endings"] = pd.to_datetime(df["Week Endings"])\ndf = df.sort_values("Week Endings").reset_index(drop=True)'),
    ('md', '## Sliding Window Dataset'),
    ('code', 'def create_dataset(dataset, look_back=1):\n    X, Y = [], []\n    for i in range(len(dataset)-look_back):\n        a = dataset[i:(i+look_back), 0]\n        X.append(a)\n        Y.append(dataset[i + look_back, 0])\n    return np.array(X), np.array(Y)\n\nlook_back = 12\nprice_data = df["Avg Ticket Price ($)"].values.astype("float32").reshape(-1, 1)\nscaler = MinMaxScaler(feature_range=(0, 1))\nprice_scaled = scaler.fit_transform(price_data)'),
    ('md', '## Pre-April 2017 Model (for 2019 event)'),
    ('code', 'cutoff_2017 = pd.to_datetime("2017-04-01")\ntrain_2017 = price_scaled[df["Week Endings"] < cutoff_2017]\nX_train1, y_train1 = create_dataset(train_2017, look_back=look_back)\nX_train1 = np.reshape(X_train1, (X_train1.shape[0], X_train1.shape[1], 1))\n\nmodel1 = keras.Sequential([\n    keras.layers.SimpleRNN(32, input_shape=(look_back, 1)),\n    keras.layers.Dense(1)\n])\nmodel1.compile(loss="mean_squared_error", optimizer="adam")\nmodel1.fit(X_train1, y_train1, epochs=20, batch_size=16, verbose=1)\nmodel1.save("model_2017.keras")'),
    ('md', '## Pre-April 2024 Model (for 2024 event)'),
    ('code', 'cutoff_2024 = pd.to_datetime("2024-04-01")\ntrain_2024 = price_scaled[df["Week Endings"] < cutoff_2024]\nX_train2, y_train2 = create_dataset(train_2024, look_back=look_back)\nX_train2 = np.reshape(X_train2, (X_train2.shape[0], X_train2.shape[1], 1))\n\nmodel2 = keras.Sequential([\n    keras.layers.SimpleRNN(32, input_shape=(look_back, 1)),\n    keras.layers.Dense(1)\n])\nmodel2.compile(loss="mean_squared_error", optimizer="adam")\nmodel2.fit(X_train2, y_train2, epochs=20, batch_size=16, verbose=1)\nmodel2.save("model_2024.keras")')
]
create_notebook('sequence-model.ipynb', seq_model_cells)

# 2. sequence-model-hypertuning.ipynb
hyper_cells = [
    ('md', '# Hyperparameter Tuning\n\nUsing Keras Tuner to find the best configuration.'),
    ('code', 'import os\nos.environ["KERAS_BACKEND"] = "torch"\nimport keras\nimport keras_tuner as kt\nimport pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import MinMaxScaler'),
    ('code', 'df = pd.read_csv("final-data.csv")\ndf["Week Endings"] = pd.to_datetime(df["Week Endings"])\ndf = df.sort_values("Week Endings").reset_index(drop=True)\n\ndef create_dataset(dataset, look_back=12):\n    X, Y = [], []\n    for i in range(len(dataset)-look_back):\n        X.append(dataset[i:(i+look_back), 0])\n        Y.append(dataset[i + look_back, 0])\n    return np.array(X), np.array(Y)\n\nprice_data = df["Avg Ticket Price ($)"].values.astype("float32").reshape(-1, 1)\nscaler = MinMaxScaler(feature_range=(0, 1))\nprice_scaled = scaler.fit_transform(price_data)\n\ncutoff = pd.to_datetime("2017-04-01")\ntrain = price_scaled[df["Week Endings"] < cutoff]\nX_train, y_train = create_dataset(train, look_back=12)\nX_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))'),
    ('code', 'def build_model(hp):\n    model = keras.Sequential()\n    model.add(keras.layers.SimpleRNN(\n        units=hp.Int("units", min_value=16, max_value=64, step=16),\n        input_shape=(12, 1)\n    ))\n    model.add(keras.layers.Dense(1))\n    model.compile(\n        optimizer=keras.optimizers.Adam(\n            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])\n        ),\n        loss="mse"\n    )\n    return model'),
    ('code', 'tuner = kt.RandomSearch(\n    build_model,\n    objective="val_loss",\n    max_trials=5,\n    executions_per_trial=1,\n    directory="tuner_dir",\n    project_name="broadway_lion_king"\n)\n\ntuner.search(X_train, y_train, epochs=10, validation_split=0.2)\n\nbest_hps = tuner.get_best_hyperparameters(num_trials=1)[0]\nprint(f"Best units: {best_hps.get(\'units\')}")\nprint(f"Best learning rate: {best_hps.get(\'learning_rate\')}")')
]
create_notebook('sequence-model-hypertuning.ipynb', hyper_cells)

# 3. causal-impact-analysis.ipynb
impact_cells = [
    ('md', '# Causal Impact Analysis\n\nComparing actual prices to counterfactual predictions for the 2019 and 2024 movie events.'),
    ('code', 'import os\nos.environ["KERAS_BACKEND"] = "torch"\nimport keras\nimport pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import MinMaxScaler\nimport pickle\nimport matplotlib.pyplot as plt\n\n# Load Data\ndf = pd.read_csv("final-data.csv")\ndf["Week Endings"] = pd.to_datetime(df["Week Endings"])\ndf = df.sort_values("Week Endings").reset_index(drop=True)\n\nprice_data = df["Avg Ticket Price ($)"].values.astype("float32").reshape(-1, 1)\nscaler = MinMaxScaler(feature_range=(0, 1))\nprice_scaled = scaler.fit_transform(price_data)'),
    ('code', '# Load Models\nmodel_2017 = keras.models.load_model("model_2017.keras")\nmodel_2024 = keras.models.load_model("model_2024.keras")\nlook_back = 12'),
    ('code', 'def create_dataset(dataset, look_back=1):\n    X, Y = [], []\n    for i in range(len(dataset)-look_back):\n        a = dataset[i:(i+look_back), 0]\n        X.append(a)\n        Y.append(dataset[i + look_back, 0])\n    return np.array(X), np.array(Y)\n\ndef evaluate_impact(model, start_date, end_date, title):\n    mask = (df["Week Endings"] >= pd.to_datetime(start_date)) & (df["Week Endings"] <= pd.to_datetime(end_date))\n    eval_df = df[mask].reset_index(drop=True)\n    \n    if len(eval_df) == 0:\n        print(f"No data for {title}")\n        return\n        \n    start_idx = df.index[mask][0]\n    input_data = price_scaled[start_idx - look_back : start_idx + len(eval_df)]\n    \n    X, y_true = create_dataset(input_data, look_back=look_back)\n    X = np.reshape(X, (X.shape[0], X.shape[1], 1))\n    \n    y_pred_scaled = model.predict(X, verbose=0)\n    y_pred = scaler.inverse_transform(y_pred_scaled)\n    y_actual = scaler.inverse_transform(y_true.reshape(-1, 1))\n    \n    avg_actual = np.mean(y_actual)\n    avg_pred = np.mean(y_pred)\n    pct_diff = ((avg_actual - avg_pred) / avg_pred) * 100\n    \n    print(f"--- {title} ---")\n    print(f"Window: {start_date} to {end_date}")\n    print(f"Actual Avg Price: ${avg_actual:.2f}")\n    print(f"Predicted Avg Price: ${avg_pred:.2f}")\n    print(f"% Difference: {pct_diff:.2f}%\\n")\n    \n    plt.figure(figsize=(10, 5))\n    plt.plot(eval_df["Week Endings"], y_actual, label="Actual Price")\n    plt.plot(eval_df["Week Endings"], y_pred, label="Predicted (Counterfactual)")\n    plt.title(f"{title} - Actual vs Counterfactual")\n    plt.legend()\n    plt.show()'),
    ('md', '## 2019 Movie Event Impact\nComparison of actual vs predicted for the 2017-2019 window.'),
    ('code', 'evaluate_impact(model_2017, "2017-04-01", "2019-07-31", "2019 Movie Impact")'),
    ('md', '## COVID Stabilization Effect\nDid the movie\'s brand awareness stabilize demand during the post-pandemic reopening?'),
    ('code', 'evaluate_impact(model_2017, "2021-08-01", "2022-12-31", "COVID Post-Opening Impact")'),
    ('md', '## 2024 Movie Event Impact\nComparison for Mufasa: The Lion King (April 2024 to Dec 2024).'),
    ('code', 'evaluate_impact(model_2024, "2024-04-01", "2024-12-31", "2024 Movie Impact")')
]
create_notebook('causal-impact-analysis.ipynb', impact_cells)
