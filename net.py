import torch
import torchvision
from torch import nn

# TODO everything lol
class PlaceholderNet(nn.Module):
	"""
	Model of a Recurrent Neural Network approach to predict the energy consumption of a building.
	"""
	def __init__(self):

		# uses parameters, methods of parent Class
		super(PlaceholderNet, self).__init__()

		# TODO add layers
		self.l1 = nn.Linear(in_features = 17, out_features = 50)
		self.l2 = nn.Linear(in_features = 50, out_features = 100)
		self.l3 = nn.Linear(in_features = 100, out_features = 1)

		

	def forward(self, x):
		"""
		defines how the data is passed from each layer to another.
		"""
		# print('')
		# x = nn.functional.relu(x)
		# print(x)
		x = nn.functional.relu(self.l1(x))
		# print(x)
		x = nn.functional.relu(self.l2(x))
		x = self.l3(x)
		# print(x)
		return x


class RecurrentPlaceholderNet(nn.Module):

	def __init__(self, input_size, hidden_size, output_size):

		super(RecurrentPlaceholderNet, self).__init__()

		self.hidden_size = hidden_size
		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(input_size+ output_size, output_size)
		self.softmax = nn.LogSoftmax(dim = 1)

	def forward(self, input, hidden):
		combined = torch.cat((input, hidden), 1)
		hidden = self.i2h(combined)
		output = self.i2o(combined)
		# output = self.softmax(output)

		return output, hidden

	def initHidden(self):
		return torch.zeros(1, self.hidden_size)


