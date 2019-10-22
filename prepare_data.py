import pandas as pd 
import numpy as np 

# loading the data
train = pd.read_csv('train.csv', index_col = False)
building_metadata = pd.read_csv('building_metadata.csv',index_col= False)
wheater = pd.read_csv('weather_train.csv', index_col = False)

# print(train.size)

columns = []
for t_c in train.columns:
	columns.append(t_c)
for b_c in building_metadata.columns:
	if b_c not in columns:
		columns.append(b_c)
for w_c in wheater.columns:
	if w_c not in columns:
		columns.append(w_c)


del wheater
print(columns)
prepared_train = pd.DataFrame(columns = columns)

# print(building_metadata[1])prepared_train
for column in train.columns:
	prepared_train[column] = train[column]


del train

site_ids = []
primary_uses = []
square_feets = []
year_builts = []
floor_count = []
i = 0
for building_id in prepared_train['building_id'].values:
	x = building_metadata.loc[building_metadata['building_id'] == building_id]
	# print(x)
	# print(building_id,x['site_id'])
	site_ids.append(x['site_id'].values.item())
	primary_uses.append(x['primary_use'].values.item())
	square_feets.append(x['square_feet'].values.item())
	year_builts.append(x['year_built'].values.item())
	floor_count.append(x['floor_count'].values.item())
	i += 1 
	if i % 10000 == 0:
		print(i)
		

prepared_train['site_id'] = site_ids
prepared_train['primary_use'] = primary_uses
prepared_train['square_feet'] = square_feet
prepared_train['year_built'] = year_builts
prepared_train['floor_count'] = floor_count

print(prepared_train)