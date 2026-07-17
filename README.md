# Lion King Movie → Broadway Ticket Price Impact

## Project Overview
This data science project measures the **causal impact of Disney Lion King movie releases on Broadway "The Lion King" musical ticket prices**. It uses counterfactual time-series forecasting: a forecasting model is trained only on pre-announcement data and used to project what prices "would have been" without the movie's influence. By comparing these counterfactual predictions against actual prices, we isolate the movie's effect on Broadway demand.

This project builds upon an existing analysis of the 2019 live-action movie and extends it by incorporating data from the December 2024 release of *Mufasa: The Lion King*.

## Motivation
Brand synergy between film releases and Broadway productions is a powerful driver of ticket sales. The goal of this analysis is to quantify that synergy for one of Broadway's most enduring shows. Specifically, we want to know:
1. Did the 2019 live-action *The Lion King* cause an immediate spike in Broadway ticket prices?
2. Did the 2019 film provide a "protective effect" that stabilized demand during the post-COVID Broadway reopening?
3. **[Original Contribution]** Did the December 20, 2024 release of *Mufasa: The Lion King* produce a similar price effect to the 2019 movie?

## Methodology
1. **Data Collection**: Weekly Broadway ticket data was collected for *The Lion King* from October 1997 through January 2026.
2. **Preprocessing**: The raw data was cleaned, nulls were handled, and the target variable (Average Ticket Price) was scaled.
3. **Sequence Modeling**: Two baseline `SimpleRNN` models were trained using Keras/PyTorch backend on a sliding window of historical prices:
    * **Model 1 (2019 Event)**: Trained on data strictly prior to April 2017 (when the 2019 movie's release date was confirmed).
    * **Model 2 (2024 Event)**: Trained on data strictly prior to April 2024 (when the *Mufasa* teaser trailer was released).
4. **Causal Impact Analysis**: The models generated counterfactual price predictions for the period following the announcements. We then compared the actual prices against the predictions.

## Key Results
Based on our trained baseline SimpleRNN, we observed the following impacts:

| Event | Analysis Window | Actual Avg Price | Predicted (Counterfactual) | % Difference |
| :--- | :--- | :--- | :--- | :--- |
| **2019 Movie Impact** | Apr 2017 - Jul 2019 | $161.05 | $156.44 | **+2.95%** |
| **COVID Stabilization** | Aug 2021 - Dec 2022 | $138.58 | $135.92 | **+1.96%** |
| **2024 Movie Impact** | Apr 2024 - Dec 2024 | $167.71 | $163.07 | **+2.85%** |

*Note: While previous studies (such as the reference project) found larger magnitudes (e.g., ~18.6% for the 2019 event), our conservative baseline model still successfully isolates a consistent, positive demand shock. Crucially, the 2024 Mufasa release generated a **+2.85%** price premium, demonstrating an effect magnitude remarkably similar to the 2019 release.*

## Technical Stack
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `keras`, `scikit-learn` (MinMaxScaler)
- **Deep Learning Backend**: `torch` (PyTorch utilized as the Keras backend due to Python 3.14 environment constraints)
- **Visualization**: `matplotlib`, `seaborn`
- **Hyperparameter Tuning**: `keras-tuner`

## Repository Structure
- `lion-king-datacollect.ipynb`: Original web scraping and basic parsing.
- `Data-analysis-initial.ipynb`: Exploratory data analysis and preprocessing.
- `sequence-model.ipynb`: Baseline SimpleRNN forecasting model construction.
- `sequence-model-hypertuning.ipynb`: Hyperparameter optimization with Keras Tuner.
- `causal-impact-analysis.ipynb`: The counterfactual predictions and final results calculation.
- `final-data.csv`: The clean, preprocessed dataset used for modeling.

## Future Work
- Incorporate exogenous variables (e.g., tourism rates, seasonality flags, Broadway macro-trends).
- Explore more complex architectures like LSTMs, GRUs, or Transformer-based time-series models.
- Conduct statistical significance tests on the difference between actual and counterfactual curves.
