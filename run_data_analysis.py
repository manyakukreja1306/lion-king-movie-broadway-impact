import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("broadway_lion_king_data.csv")

# Starting EDA
print(data.shape)

list(data.columns)


print(data.to_string())


data.info()


for col in data.columns:
    print(data[col].value_counts())

for col in data.columns:
    print(data[col].value_counts(dropna = False))
    print(data[col].value_counts(dropna = False).get(np.nan, 0))
    print(data[col].value_counts(dropna = False).get('', 0))

# Data Transformation start
data['Week Endings'] = pd.to_datetime(
    data['Week Endings'],
    format="%b %d, %Y",
    errors='raise'
)
print(data['Week Endings'])


data['Avg Ticket Price ($)'] = data['Avg Ticket Price ($)'].apply(lambda x: float(x[1:]))
print(data['Avg Ticket Price ($)'])

# Using .map() method to do data transformation for 'Top Ticket Price ($)' column since it allows an additional argument
# na_action to ignore NaN values as they are encountered
data['Top Ticket Price ($)'] = data['Top Ticket Price ($)'].map(lambda x: float(x[1:]), na_action = 'ignore')
print(data['Top Ticket Price ($)'])

data['Seats Sold'] = data['Seats Sold'].fillna(0).astype(int)
print(data['Seats in Theatre'])


data['% Cap'] = (
    data['% Cap']
    .astype(str)              # defensive: handles mixed types
    .str.replace('%', '', regex=False)
    .astype(float)
)


print(data.info())

sorted_df = data.sort_values(by='Week Endings', ascending=False)
print(sorted_df[sorted_df['Seats in Theatre'] == 0])


data = data[~data['Week Endings'].isin(pd.to_datetime([
    '2007-11-18',
    '2007-11-25'
]))]

print(data.shape)


# Confirming if data has been dropped accurately
print(data[data['Seats in Theatre'] == 0].shape)

dates = data['Week Endings']
grouped_df = data.groupby(by=dates.dt.year)
mean_series = grouped_df['Top Ticket Price ($)'].mean()
print(mean_series)

# Setting the top ticket price for year 2021:
# This will be done manually bcuz there exists no mean for the year 2021, all values are NaN

# Setting the boolean mask first only, as we will be reusing it
boolean_mask = np.isnan(data['Top Ticket Price ($)']) & (data['Week Endings'].dt.year == 2021)
# Checking before assigning mean value:
print(data['Top Ticket Price ($)'][boolean_mask].head(3))
# Assigning mean value:
data.loc[boolean_mask, 'Top Ticket Price ($)'] = 199.0
# After data cleaning:
data['Top Ticket Price ($)'][boolean_mask].head(3)

data.groupby(data['Week Endings'].dt.year)['Top Ticket Price ($)'].apply(lambda x: x.isna().sum())


def set_top_t_price_to_mean(weekdata):
    mask = weekdata['Top Ticket Price ($)'].isna()
    years = weekdata.loc[mask, 'Week Endings'].dt.year
    weekdata.loc[mask, 'Top Ticket Price ($)'] = years.map(mean_series)
    return weekdata


data = set_top_t_price_to_mean(data)


print(data[np.isnan(data['Top Ticket Price ($)'])])

data.info()

data.describe().transpose()

sns.set_style("whitegrid")
sns.histplot(data, x='Avg Ticket Price ($)', stat='density', kde=True, kde_kws=dict(cut=3))

# Some basic plotting to check data scale to determine whether data needs to be normalised or not
# before being used with the sequence model for prediction
time = data['Week Endings']
plt.figure(figsize=(8, 8))
plt.grid(True)
plt.plot(time, data['Avg Ticket Price ($)'], color='b', label="Average Ticket Price")
plt.plot(time, data['Top Ticket Price ($)'], color='r', label="Top Ticket Price")
plt.legend(loc=1)
# plt.show()

data.to_csv('final-data.csv', index=False)



