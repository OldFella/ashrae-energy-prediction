import pandas as pd 
import numpy as np 
import math

train = pd.read_csv('train.csv')
buildings = pd.read_csv('building_metadata.csv')
weather = pd.read_csv('weather_train.csv')
usage = pd.read_csv('primary_usage_translations.csv')


building_ids = buildings['building_id'].tolist()

result = pd.DataFrame(columns = [	'building_id', 'meter' ,'primary_usage', 'square_feet', 'year_built', 'floor_count', 'site_id', 'air_temperature',
								 	'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed',
								 	'month', 'day', 'hour'])

result['building_id'] = train['building_id']
result['meter'] = train['meter']
# print(result)

for i, row in enumerate(buildings.iterrows(),0):
	row = row[1]
	building_id = row['building_id']
	# print(building_id)
	# print(row)
	primary_usage = usage.loc[usage['primary_usage'] == row['primary_use']]
	u = primary_usage['index'].values.item()
	# print(u)
	new_row = row.tolist()
	# building_id = 
	site_id = new_row[0]
	new_row = new_row[2:]
	new_row[0] = u
	new_row.append(site_id)
	# for x in range(len(new_row)):
	# 	if math.isnan(new_row[x]):
	# 		new_row[x] = -1
	# print(new_row)
	result.ix[result['building_id'] == building_id, ['primary_usage', 'square_feet', 'year_built', 'floor_count', 'site_id']] = new_row
	print(new_row)
	print(i)

result = result.fillna(value = -1)
# x = result.loc[result['building_id'] == 0]
print(result)

result.to_csv('prepared_train.csv')