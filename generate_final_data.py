import pandas as pd 
import numpy as np 

data = pd.read_csv("broadway_lion_king_data.csv")

data['Week Endings'] = pd.to_datetime(
    data['Week Endings'],
    format="%b %d, %Y",
    errors='raise'
)
data['Avg Ticket Price ($)'] = data['Avg Ticket Price ($)'].apply(lambda x: float(x[1:]))
data['Top Ticket Price ($)'] = data['Top Ticket Price ($)'].map(lambda x: float(x[1:]), na_action = 'ignore')
data['Seats Sold'] = data['Seats Sold'].fillna(0).astype(int)
data['% Cap'] = (
    data['% Cap']
    .astype(str)
    .str.replace('%', '', regex=False)
    .astype(float)
)

data = data[~data['Week Endings'].isin(pd.to_datetime([
    '2007-11-18',
    '2007-11-25'
]))]

dates = data['Week Endings']
grouped_df = data.groupby(by=dates.dt.year)
mean_series = grouped_df['Top Ticket Price ($)'].mean()

boolean_mask = np.isnan(data['Top Ticket Price ($)']) & (data['Week Endings'].dt.year == 2021)
data.loc[boolean_mask, 'Top Ticket Price ($)'] = 199.0

def set_top_t_price_to_mean(weekdata):
    mask = weekdata['Top Ticket Price ($)'].isna()
    years = weekdata.loc[mask, 'Week Endings'].dt.year
    weekdata.loc[mask, 'Top Ticket Price ($)'] = years.map(mean_series)
    return weekdata

data = set_top_t_price_to_mean(data)
data.to_csv('final-data.csv', index=False)
print("Data generation complete.")
