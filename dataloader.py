from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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
					 	cloud_coverage,dew_temperature, precip_depth_1_hr, sea_level_pressure, wind_direction, wind_speed]
			
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
		wheater_data = wheater_data[0]

		# transform the time form str to int and add individualy day, month and hour
		time = wheater_data[1]
		time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
		day = time.day
		hour = time.hour
		month = time.month
		wheater_data = np.delete(wheater_data,1)
		time = np.array([day,hour,month])
		wheater_data = np.concatenate([wheater_data, time])
		data = np.concatenate([data, wheater_data])


		#transform output to tensors
		data = data.astype(float)
		data = torch.Tensor(data)
		target = torch.Tensor(np.array([target]))
		return data, target
