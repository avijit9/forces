import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pdb

class RGBMaskEncoderCNN(nn.Module):
	def __init__(self):
		super(RGBMaskEncoderCNN, self).__init__()

		self.alexnet = models.alexnet(pretrained=True)
		new_classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:-1])
		self.alexnet.classifier = new_classifier

		# get the pre-trained weights of the first layer
		pretrained_weights = self.alexnet.features[0].weight
		new_features = nn.Sequential(*list(self.alexnet.features.children()))
		new_features[0] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
		# For M-channel weight should randomly initialized with Gaussian
		new_features[0].weight.data.normal_(0, 0.001)
		# For RGB it should be copied from pretrained weights
		new_features[0].weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
		self.alexnet.features = new_features

	def forward(self, images):
		"""Extract Feature Vector from Input Images"""
		features = self.alexnet(images)
		return features




class ForceEncoderCNN(nn.Module):
	def __init__(self):
		super(ForceEncoderCNN, self).__init__()
		self.alexnet = models.alexnet(pretrained=True)
		new_classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:-1])
		self.alexnet.classifier = new_classifier

		

	def forward(self, images):
		"""Extract Feature Vector from Input Images"""
		features = self.alexnet(images)
		return features


class DecoderRNN(nn.Module):
	
	def __init__(self, args, device):
		super(DecoderRNN, self).__init__()

		self.input_size = args.input_size
		self.hidden_size = args.hidden_size
		self.output_size = args.output_size
		self.seq_len = args.seq_len
		self.num_layers = args.num_layers
		self.dropout = args.dropout
		self.batch_size = args.batch_size

		# self.rnn_layer = nn.RNN(self.input_size, self.hidden_size, self.num_layers,
		#  nonlinearity = 'relu', dropout = self.dropout, batch_first = True)
		self.rnn_layer = nn.GRU(self.input_size, self.hidden_size, self.num_layers,
			dropout = self.dropout, batch_first = True)

		self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, force_feature, rgb_mask_feature):

		I = torch.cat((force_feature, rgb_mask_feature), dim = 1)[:, None, :] # makes it (batch, 1, input_size)
		I = I.repeat((1, self.seq_len, 1))
		# no need to initialize hidden state to zero as this is already done in RNN (pytorch you beauty :P)
		rnn_output, h_n = self.rnn_layer(I)
		
		linear_input = rnn_output.contiguous().view(force_feature.shape[0] * self.seq_len, -1)
		output = self.hidden_to_output(linear_input)
		return output





		


		



		

		




