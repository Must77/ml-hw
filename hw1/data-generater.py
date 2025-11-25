import pandas as pd
import numpy as np

np.random.seed(0)
data = {
   'Id': range(1, 101),
   'LotArea': np.concatenate([np.random.normal(10000, 2000, 95), [-500, 50000, 60000, np.nan, 70000]]),
   'YearBuilt': np.random.randint(1950, 2021, 100),
   'BedroomAbvGr': np.concatenate([np.random.randint(1, 6, 98), [np.nan, 10]]),
   'KitchenQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po', np.nan], 100, p=[0.1, 0.2, 0.5, 0.15, 0.04, 0.01]),
   'GarageCars': np.concatenate([np.random.choice([0,1,2,3], 97), [np.nan, np.nan, 10]]),
   'PoolArea': np.concatenate([np.zeros(90), np.random.choice([50, 100], 9), [np.nan]]),
   'MiscFeature': np.random.choice([np.nan, 'Shed', 'TenC', 'Elev'], 100, p=[0.85, 0.1, 0.03, 0.02]),
   'SalePrice': np.concatenate([np.random.normal(200000, 50000, 97), [1e6, np.nan, 50000]])
}
df = pd.DataFrame(data)
df.to_csv('house_data.csv', index=False)