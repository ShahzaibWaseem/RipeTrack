from math import sqrt
import numpy as np

import torch
import torch.nn as nn

from models.gnn import GraphNet
from models.gcn import GNN
from models.resblock import conv3x3

from utils_graph import prepare_data
class Network(nn.Module):
	def __init__(self, block, block_num, input_channel, output_channel, fusion="concat"):
		super(Network, self).__init__()

		self.in_channels = input_channel
		self.out_channels = output_channel
		self.fusion = fusion
		self.input_conv = conv3x3(self.in_channels, out_channels=64)
		self.input_relu = nn.ReLU()
		self.img_size = 512//4

		self.downsampling_block = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1),
												nn.BatchNorm2d(64),
												nn.ReLU(),
												nn.Conv2d(64, 64, 3, stride=2, padding=1),
												nn.BatchNorm2d(64),
												nn.ReLU())

		self.gnn_block = nn.Sequential(nn.BatchNorm2d(64),
									   GraphNet(img_size=self.img_size, input_channel=64, pred_edge=True),
									   nn.BatchNorm2d(64))

		self.upsampling_block = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
											  nn.BatchNorm2d(64),
											  nn.ReLU(),
											  nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
											  nn.BatchNorm2d(64),
											  nn.ReLU())

		self.conv_seq = self.make_layer(block, block_num, 64, 64, stride=1, cardinality=32, base_width=4, widen_factor=1)
		self.conv = conv3x3(64, 64)
		self.relu = nn.ReLU(inplace=True)

		# if (self.fusion == "concat"):
		# 	self.output_conv = conv3x3(in_channels=64*2, out_channels=self.out_channels)
		# elif (self.fusion == "multiply" or self.fusion == "add" or self.fusion == "resnext"):
		self.output_conv = conv3x3(in_channels=64, out_channels=self.out_channels)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
				m.weight.data.normal_(0,sqrt(2./n))

	def make_layer(self, block, num_layers, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
		layers = []
		for _ in range(num_layers):
			layers.append(block(in_channels, out_channels, stride, cardinality, base_width, widen_factor))
		return nn.Sequential(*layers)

	def split_tensor(self, image, split_size=64):
		""" splits the tensor if the model is too big for the GPU's VRAM """
		torch.cuda.empty_cache()
		size = image.size(2)
		output = torch.zeros(image.shape, dtype=torch.float32).cuda()
		for split_i in range(size//split_size):
			for split_j in range(size//split_size):
				start_i, end_i = split_i*split_size, (split_i+1)*split_size
				start_j, end_j = split_j*split_size, (split_j+1)*split_size
				gnn_out = self.gnn_block(self.downsampling_block(image[:, :, start_i:end_i, start_j:end_j]))
				output[:, :, start_i:end_i, start_j:end_j] = self.upsampling_block(gnn_out)
		return output

	def forward(self, x):
		self.img_size = x.shape[2]//4
		out = self.input_conv(x)
		out = self.input_relu(out)
		# residual = out

		# if x.shape[2] > 64:
		# 	gnn_out = self.split_tensor(out)
		# else:
		# print(out.shape)

		gnn_out = self.downsampling_block(out)
		# # print(gnn_out.shape)

		gnn_out = self.gnn_block(gnn_out)
		# # print(gnn_out.shape)

		gnn_out = self.upsampling_block(gnn_out)
		# print(gnn_out.shape)

		# out = self.conv_seq(out)
		# out = self.conv(out)
		# out = torch.add(out, residual)
		# out = self.relu(out)

		# if (self.fusion == "concat"):
		# 	out = torch.cat((out, gnn_out), 1)
		# elif (self.fusion == "add"):
		# 	out = out + gnn_out
		# elif (self.fusion == "multiply"):
		# 	out = out * gnn_out

		out = self.output_conv(gnn_out)
		# print(out.shape)

		return out