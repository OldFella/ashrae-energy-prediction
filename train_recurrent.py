import torch
from net import RecurrentPlaceholderNet
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from dataset import AshraeDataset
import copy
import numpy as np


def train(model, criterion, lr = 0.007, epochs = 1000):

	print('data is loading...')
	dataset = AshraeDataset('train.csv', 'building_metadata.csv', 'weather_train.csv')
	print('data is finished loading.')
	optimizer = optim.Adam(model.parameters(), lr = lr)
	
	for epoch in range(epochs):

		print('epoch {}'.format(epoch))
		for i in range(1):
			optimizer.zero_grad()

			print('current building data is loading...')
			current_data, current_target = dataset.getbuildingdata(0, random_start = True, length = 100)
			print('current building data is finished loading.')
			hidden = None
			if len(current_data) > 9000:
				break

			else:
				# for x in range(current_data.size()[0]):
					# if x % 100 == 99:
						# print(x)
					# print(current_data[x])
				if hidden == None:

					output, hidden = model(current_data)

				else:
					output, hidden = model(current_data, hidden)
				loss = criterion(output, current_target)

				loss.backward()
				optimizer.step()

				# for p in model.parameters():
				# 	p.data.add_(-lr, p.grad.data)


				print(loss.item())
	current_data, target = dataset.getbuildingdata(0)
	output, hidden = model(current_data)
	loss = criterion(output, target)
	print(output,loss.item())

train(nn.LSTM(17,1, num_layers= 10), nn.MSELoss())