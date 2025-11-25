import pandas as pd
import numpy as np

np.random.seed(0)

ids = range(1, 101)

lotarea_main = np.random.normal(10000, 2000, 95)
lotarea_tail = np.array([-500, 50000, 60000, np.nan, 70000])
lotarea = np.concatenate([lotarea_main, lotarea_tail])

year_built = np.random.randint(1950, 2021, 100)

bedroom_main = np.random.randint(1, 6, 98)
bedroom_tail = np.array([np.nan, 10])
bedrooms = np.concatenate([bedroom_main, bedroom_tail])

kitchen_qual = np.random.choice(
    ['Ex', 'Gd', 'TA', 'Fa', 'Po', np.nan],
    100,
    p=[0.1, 0.2, 0.5, 0.15, 0.04, 0.01],
)

garage_main = np.random.choice([0, 1, 2, 3], 97)
garage_tail = np.array([np.nan, np.nan, 10])
garage_cars = np.concatenate([garage_main, garage_tail])

pool_area = np.concatenate([np.zeros(90), np.random.choice([50, 100], 9), [np.nan]])

misc_feature = np.random.choice(
    [np.nan, 'Shed', 'TenC', 'Elev'],
    100,
    p=[0.85, 0.1, 0.03, 0.02],
)

saleprice_main = np.random.normal(200000, 50000, 97)
saleprice_tail = np.array([1e6, np.nan, 50000])
saleprice = np.concatenate([saleprice_main, saleprice_tail])

data = {
    'Id': ids,
    'LotArea': lotarea,
    'YearBuilt': year_built,
    'BedroomAbvGr': bedrooms,
    'KitchenQual': kitchen_qual,
    'GarageCars': garage_cars,
    'PoolArea': pool_area,
    'MiscFeature': misc_feature,
    'SalePrice': saleprice,
}

df = pd.DataFrame(data)
df.to_csv('house_data.csv', index=False)