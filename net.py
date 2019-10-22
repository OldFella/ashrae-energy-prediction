import torch
import torchvision
from torch import nn

# TODO everything lol
class EnergyNet(torch.nn.Module):
	"""
	Model of a Recurrent Neural Network approach to predict the energy consumption of a building.
	"""
	def __init__(self):

		# uses parameters, methods of parent Class
		super(EnergyNet, self).__init__()

		# TODO add layers
		...

	def forward(self, x):
		"""
		defines how the data is passed from each layer to another.
		"""

		return x