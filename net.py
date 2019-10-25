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
		self.l2 = nn.Linear(in_features = 50, out_features = 1)

		

	def forward(self, x):
		"""
		defines how the data is passed from each layer to another.
		"""
		# print('')
		# x = nn.functional.relu(x)
		# print(x)
		x = self.l1(x)
		# print(x)
		x = self.l2(x)
		# print(x)
		return x