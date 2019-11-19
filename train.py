import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from torch import nn
# from prepare_data import training_loader, create_loader, transform, validation_loader
from net import PlaceholderNet

from dataset import AshraeDataset
# from _config import MODEL_PATH
import copy
import numpy as np
# from utils import print_time, progress
# import visualize
# from test import test



def train(model = None, criterion = None, training_epochs = 1, batch_size = 1, learning_rate = 0.001):
	""" 
	Runs through the training data, makes a prediction and computes loss, then backpropagates
	the result through the model and adjusts the weights and biases until a local minimum in the loss
	function is reached.
	"""

	# optimizer searches fo a local minimum of in the lossfunction with different input parameters
	optimizer = optim.Adam(model.parameters(), lr = learning_rate)
	# graph_loss = []
	# graph_accuracy = [(0,0)]
	# graph_validation_loss = []

	best_model = None

	dataset = AshraeDataset('train.csv', 'building_metadata.csv', 'weather_train.csv')

	threshold = 0

	for epoch in range(training_epochs):
		running_loss = 0.0
		training_loader = DataLoader(dataset, batch_size = batch_size)
		average_loss = 0
		print('')
		number_of_files = len(training_loader.dataset)
		print(len(training_loader))
		for i, data in enumerate(training_loader, 0):
			if i == 10:
				break

			# get input for training
			inputs, labels = data
			# print(inputs, labels)

			# init optimizer with 0
			optimizer.zero_grad()

			# rung data trough net
			outputs = model(inputs)
			# print(outputs)

			# compute loss (compare output to label)
			loss = criterion(outputs, labels)

			# backpropagate loss
			loss.backward()

			# tweak parameters
			optimizer.step()

			# add loss to overall loss
			running_loss += loss.item()

			# pretty print progress
			if i % 10 == 9:  # append a the average of the last 10 losses as point to the loss/epoch graph_loss
				average_loss = running_loss/10
				print(i, average_loss)
				# graph_loss.append((epoch + i/(number_of_files/batch_size), average_loss))
				running_loss = 0.0

	# 		# Progress bar
	# 		progress(i, number_of_files/batch_size, epoch + 1, '{}/{:.0f} Loss: {:.2f}'.format(i, number_of_files/batch_size, average_loss))

	# 	# Validate the result of the epoch
	# 	test_loss, correct, dataset_size, accuracy_percent = test(validation_loader, model)
	# 	graph_accuracy.append((epoch + 1, accuracy_percent/100))
	# 	graph_validation_loss.append((epoch + 1, test_loss))

	# 	if accuracy_percent > threshold:
	# 		best_model = copy.deepcopy(model)
	# 		threshold = accuracy_percent

	# model_name = '{}epochs_{}learingrate_{}batchsize.pt'.format(training_epochs, learning_rate, batch_size)
	# torch.save(best_model, MODEL_PATH + model_name)

	# print("\nmodel: " + model_name + " has been saved.")
	# print("Best Percentage: {:.2f}".format(threshold))
	return best_model #, (graph_loss, graph_accuracy, graph_validation_loss)


train(	model = PlaceholderNet(),
		criterion = nn.MSELoss())