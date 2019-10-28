from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math
from datetime import datetime

train = 'train.csv'
buildings = 'building_metadata.csv'
wheater = 'weather_train.csv'

class AshraeDataset(Dataset):
	"""Ashrae Energy Prediction Dataset """
	def __init__(self, csv_file, csv_buildings, csv_wheater):

		self.energy_frame = pd.read_csv(csv_file, index_col = False)
		self.building_metadata = pd.read_csv(csv_buildings, index_col = False)
		self.wheater = pd.read_csv(csv_wheater, index_col = False)
		self.primary_usage = pd.read_csv('primary_usage_translations.csv',index_col = False)

	def __len__(self):
		return len(self.energy_frame)

	def __getitem__(self, idx):
		"""
		returns 

			the input variables:
			data = [	Building_id, meter, primary_usage, square_feet, year_built, floor_count, site_id, air_temperature,
					 	cloud_coverage,dew_temperature, precip_depth_1_hr, sea_level_pressure, wind_direction, wind_speed, month, day, hour]
			
			target = meter_reading
		"""
		data = self.energy_frame.iloc[idx]
		target = data['meter_reading']
		building_id = data['building_id']
		date = data['timestamp']
		data = data.values

		# Drop the timestamp to avoid duplicates and meter_reading because its the target
		data = data[:-2]
		
		#get building_metadata and drop the site_id and building_id to avoid duplicates
		building_data = self.building_metadata.loc[building_id]
		building_data = building_data.values
		site_id = building_data[0]
		building_data = building_data[2:]
		usage = building_data[0]
		usage_index = self.primary_usage.loc[self.primary_usage['primary_usage'] == usage]
		usage_index = usage_index['index'].values.item()
		building_data[0] = usage_index
		data = np.concatenate([data, building_data])

		#get the row with the current site_id and timestamp from wheater data  
		wheater_data = self.wheater.loc[(self.wheater['site_id'] == site_id) & (self.wheater['timestamp'] == date)]
		wheater_data = wheater_data.values
		if len(wheater_data) == 0:
			wheater_data = [-1,-1,-1,-1,-1,-1,-1,-1,-1]
		else: 
			wheater_data = wheater_data[0]

		# transform the time form str to int and add individualy day, month and hour
		time = date
		time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
		day = time.day
		hour = time.hour
		month = time.month
		wheater_data = np.delete(wheater_data,1)
		time = np.array([month,day,hour])
		wheater_data = np.concatenate([wheater_data, time])
		data = np.concatenate([data, wheater_data])
		# print(data)
		data = data.astype(float)
		for i in range(len(data)):
			# print(type(data[i]), data[i])
			if math.isnan(data[i]):
				# print('nan--')
				data[i] = -1
		#transform output to tensors
		data = torch.Tensor(data)
		target = torch.Tensor(np.array([target]))
		# target = target.astype(np.int_)
		return data, target


	def __getbuildingitem__(self,idx,b_id):
		building_id_list = self.energy_frame['building_id'].values
		idxs = self.energy_frame.index[self.energy_frame['building_id'] == b_id].tolist()
		return self.__getitem__(idxs[idx])

	def getbuildingdata(self,b_id, random_start = False, length = -1):

		building_id_list = self.energy_frame['building_id'].values
		idxs = self.energy_frame.index[self.energy_frame['building_id'] == b_id].tolist()
		sequence_length = len(idxs)
		# print(sequence_length)
		start = 0
		if random_start:
			start = np.random.randint(0,sequence_length)
		data = []
		target = []
		sequence_end = start + length
		for building in range(len(idxs)):
			current_building = building + start
			if current_building == sequence_end or current_building >= len(idxs):
				break
			current_data, current_target = self.__getitem__(idxs[current_building])
			data.append([current_data.numpy()])
			target.append([current_target.numpy()])
			# print(data, target)
		return torch.Tensor(np.array(data)), torch.Tensor(np.array(target))


# d = AshraeDataset(train, buildings, wheater)

# # for x in range(1000):
# print('-')
# x = d.getbuildingdata(1,random_start = True, length = 10000000000)
# print(x[0][0], len(x[0]))