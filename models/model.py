from math import sqrt

import torch
import torch.nn as nn

from models.gnn import GraphNet
from models.resblock import conv3x3

class Network(nn.Module):
	def __init__(self, block, block_num, input_channel, output_channel, fusion="concat"):
		super(Network, self).__init__()

		self.in_channels = input_channel
		self.out_channels = output_channel
		self.fusion = fusion
		self.input_conv = conv3x3(self.in_channels, out_channels=64)
		self.input_relu = nn.ReLU()

		self.downsampling_block = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1),
												nn.BatchNorm2d(64),
												nn.ReLU(),
												nn.Conv2d(64, 64, 3, stride=2, padding=1),
												nn.BatchNorm2d(64),
												nn.ReLU())

		self.gnn_block = nn.Sequential(nn.BatchNorm2d(64),
									   GraphNet(img_size=64, input_channel=64, pred_edge=True),
									   nn.BatchNorm2d(64))

		self.upsampling_block = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
											  nn.BatchNorm2d(64),
											  nn.ReLU(),
											  nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
											  nn.BatchNorm2d(64),
											  nn.ReLU())

		self.conv_seq = self.make_layer(block, block_num)
		self.conv = conv3x3(64, 64)
		self.relu = nn.ReLU(inplace=True)

		if (self.fusion == "concat"):
			self.output_conv = conv3x3(in_channels=64*2, out_channels=self.out_channels)
		elif (self.fusion == "multiply" or self.fusion == "add"):
			self.output_conv = conv3x3(in_channels=64, out_channels=self.out_channels)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
				m.weight.data.normal_(0,sqrt(2./n))

	def make_layer(self, block, num_layers):
		layers = []
		for _ in range(num_layers):
			layers.append(block())
		return nn.Sequential(*layers)

	def split_tensor(self, image, split_size=64):
		""" splits the tensor if the model is too big for the GPU's VRAM """
		torch.cuda.empty_cache()
		size = image.size(2)
		output = torch.zeros(image.shape, dtype=torch.float32).cuda()
		for split in range(size//split_size):
			start, end = split*split_size, (split+1)*split_size
			output[:, :, start:end, start:end] = self.gnn_block(image[:, :, start:end, start:end])
		return output

	def forward(self, x):
		out = self.input_conv(x)
		out = self.input_relu(out)
		residual = out

		gnn_out = self.downsampling_block(out)
		gnn_out = self.gnn_block(gnn_out)
		gnn_out = self.upsampling_block(gnn_out)

		out = self.conv_seq(out)
		out = self.conv(out)
		out = torch.add(out, residual)
		out = self.relu(out)

		if (self.fusion == "concat"):
			out = torch.cat((out, gnn_out), 1)
		elif (self.fusion == "add"):
			out = out + gnn_out
		elif (self.fusion == "multiply"):
			out = out * gnn_out

		out = self.output_conv(out)
		return out