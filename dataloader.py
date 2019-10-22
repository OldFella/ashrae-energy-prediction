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

	def __len__(self):
		return len(self.energy_frame)

	def __getitem__(self, idx):

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
		data = np.concatenate([data, building_data])

		#get wheater data and transform the timestamp string to integers 
		wheater_data = self.wheater.loc[(self.wheater['site_id'] == site_id) & (self.wheater['timestamp'] == date)]
		wheater_data.drop(columns = ['timestamp'])
		wheater_data = wheater_data.values
		wheater_data = wheater_data[0]
		time = wheater_data[1]
		time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
		day = time.day
		hour = time.hour
		month = time.month
		wheater_data = np.delete(wheater_data,1)
		time = np.array([day,hour,month])
		wheater_data = np.concatenate([wheater_data, time])
		data = np.concatenate([data, wheater_data])


		#TODO transform output to tensors

		return data, target
