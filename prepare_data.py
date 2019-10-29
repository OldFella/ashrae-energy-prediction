import pandas as pd 
import numpy as np 
import math
import time
from datetime import datetime


start = datetime.now()

# loading data
train = pd.read_csv('train.csv')
buildings = pd.read_csv('building_metadata.csv')
weather = pd.read_csv('weather_train.csv')
usage = pd.read_csv('primary_usage_translations.csv')


# merge train.csv and building_metadata.csv
result = pd.merge(train, buildings, on = 'building_id')


# merge result and weather
result = pd.merge(result, weather, on = ['timestamp', 'site_id'])

# clear RAM of finished data
del train, buildings, weather

# create dict to translate primary_use form str to int
usage_dict = usage.set_index('primary_usage').to_dict()['index']
del usage

# translate primary_use
primary_use = result['primary_use']
translated_usage =  primary_use.map(lambda x: usage_dict[x])
result['primary_use'] = translated_usage



# translate timestamp from str to individual rows for month, day and hour
timestamp = pd.to_datetime(result['timestamp'])

hours = timestamp.dt.hour
day = timestamp.dt.day
month = timestamp.dt.month

result['month'] = month
result['day'] = day
result['hour'] = hours


# remove useless column
del result['timestamp']


print('replacing nans...')
result = result.fillna(value = -1)

print('sorting dataframe...')
result = result.sort_values(by = ['hour', 'day', 'month', 'building_id'])

print('saving to csv...')
result.to_csv('prepared_train.csv',index = False)

print('overall execution time: {}'.format(datetime.now() - start))