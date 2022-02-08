from math import sqrt

import torch
import torch.nn as nn
from models.resblock import conv3x3

class Network(nn.Module):
	def __init__(self, block, block_num=10, input_channel=4, n_hidden=64, output_channel=51):
		super(Network, self).__init__()
		self.input_conv = conv3x3(input_channel, out_channels=n_hidden)
		self.input_relu = nn.ReLU()
		self.conv_seq = self.make_layer(block, block_num, n_hidden, n_hidden, stride=1, cardinality=32, base_width=4, widen_factor=1)
		self.conv = conv3x3(n_hidden, n_hidden)
		self.relu = nn.ReLU(inplace=True)
		self.output_conv = conv3x3(in_channels=n_hidden, out_channels=output_channel)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
				m.weight.data.normal_(0,sqrt(2./n))

	def make_layer(self, block, num_layers, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
		layers = []
		for _ in range(num_layers):
			layers.append(block(in_channels, out_channels, stride, cardinality, base_width, widen_factor))
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.input_conv(x)
		out = self.input_relu(out)
		residual = out

		out = self.conv_seq(out)
		out = self.conv(out)
		out = torch.add(out, residual)
		out = self.relu(out)

		out = self.output_conv(out)
		return out