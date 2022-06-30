from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

def convolution_3(in_channels, out_channels):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

class resblock(nn.Module):
	def __init__(self):
		super(resblock, self).__init__()
		self.conv1 = convolution_3(64, 64)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = convolution_3(64, 64)
		self.bn2 = nn.BatchNorm2d(64)

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = torch.mul(out, 0.1)
		out = torch.add(out,residual)
		return out

class ResNeXtBottleneck(nn.Module):
	"""
	RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
	"""
	def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
		"""
			in_channels: input channel dimensionality
			out_channels: output channel dimensionality
			stride: conv stride. Replaces pooling layer.
			cardinality: num of convolution groups.
			base_width: base number of channels in each group.
			widen_factor: factor to reduce the input dimensionality before convolution.
		"""
		super(ResNeXtBottleneck, self).__init__()
		width_ratio = out_channels / (widen_factor * 64.)
		D = cardinality * int(base_width * width_ratio)
		self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn_reduce = nn.BatchNorm2d(D)
		self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
		self.bn = nn.BatchNorm2d(D)
		self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn_expand = nn.BatchNorm2d(out_channels)

		self.shortcut = nn.Sequential()
		if in_channels != out_channels:
			self.shortcut.add_module("shortcut_conv",
									 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
											   bias=False))
			self.shortcut.add_module("shortcut_bn", nn.BatchNorm2d(out_channels))

		self.apply(self._init_weights)

	def _init_weights(self, module):
		if(isinstance(module, nn.Conv2d)):
			nn.init.xavier_uniform_(module.weight)
		elif isinstance(module, nn.BatchNorm2d):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)

	def forward(self, x):
		bottleneck = self.conv_reduce.forward(x)
		bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
		bottleneck = self.conv_conv.forward(bottleneck)
		bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
		bottleneck = self.conv_expand.forward(bottleneck)
		bottleneck = self.bn_expand.forward(bottleneck)
		residual = self.shortcut.forward(x)
		return F.relu(residual + bottleneck, inplace=True)