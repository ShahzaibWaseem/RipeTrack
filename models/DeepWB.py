"""
 Main blocks of the network
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import torch
import torch.nn as nn

class DoubleConvBlock(nn.Module):
	"""double conv layers block"""
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)

class DownBlock(nn.Module):
	"""Downscale block: maxpool -> double conv block"""
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConvBlock(in_channels, out_channels))

	def forward(self, x):
		return self.maxpool_conv(x)

class BridgeDown(nn.Module):
	"""Downscale bottleneck block: maxpool -> conv"""
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.maxpool_conv(x)

class BridgeUP(nn.Module):
	"""Downscale bottleneck block: conv -> transpose conv"""
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv_up = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
		)

	def forward(self, x):
		return self.conv_up(x)

class UpBlock(nn.Module):
	"""Upscale block: double conv block -> transpose conv"""
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv = DoubleConvBlock(in_channels * 2, in_channels)
		self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

	def forward(self, x1, x2):
		x = torch.cat([x2, x1], dim=1)
		x = self.conv(x)
		return torch.relu(self.up(x))

class OutputBlock(nn.Module):
	"""Output block: double conv block -> output conv"""
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.out_conv = nn.Sequential(
			DoubleConvBlock(in_channels * 2, in_channels),
			nn.Conv2d(in_channels, out_channels, kernel_size=1))

	def forward(self, x1, x2):
		x = torch.cat([x2, x1], dim=1)
		return self.out_conv(x)

class deepWBNet(nn.Module):
	def __init__(self):
		super(deepWBNet, self).__init__()
		self.n_channels = 3
		self.encoder_inc = DoubleConvBlock(self.n_channels, 24)
		self.encoder_down1 = DownBlock(24, 48)
		self.encoder_down2 = DownBlock(48, 96)
		self.encoder_down3 = DownBlock(96, 192)
		self.encoder_bridge_down = BridgeDown(192, 384)
		self.awb_decoder_bridge_up = BridgeUP(384, 192)
		self.awb_decoder_up1 = UpBlock(192, 96)
		self.awb_decoder_up2 = UpBlock(96, 48)
		self.awb_decoder_up3 = UpBlock(48, 24)
		self.awb_decoder_out = OutputBlock(24, self.n_channels)
		self.tungsten_decoder_bridge_up = BridgeUP(384, 192)
		self.tungsten_decoder_up1 = UpBlock(192, 96)
		self.tungsten_decoder_up2 = UpBlock(96, 48)
		self.tungsten_decoder_up3 = UpBlock(48, 24)
		self.tungsten_decoder_out = OutputBlock(24, self.n_channels)
		self.shade_decoder_bridge_up = BridgeUP(384, 192)
		self.shade_decoder_up1 = UpBlock(192, 96)
		self.shade_decoder_up2 = UpBlock(96, 48)
		self.shade_decoder_up3 = UpBlock(48, 24)
		self.shade_decoder_out = OutputBlock(24, self.n_channels)

	def forward(self, x):
		x1 = self.encoder_inc(x)
		x2 = self.encoder_down1(x1)
		x3 = self.encoder_down2(x2)
		x4 = self.encoder_down3(x3)
		x5 = self.encoder_bridge_down(x4)
		x_awb = self.awb_decoder_bridge_up(x5)
		x_awb = self.awb_decoder_up1(x_awb, x4)
		x_awb = self.awb_decoder_up2(x_awb, x3)
		x_awb = self.awb_decoder_up3(x_awb, x2)
		awb = self.awb_decoder_out(x_awb, x1)
		x_t = self.tungsten_decoder_bridge_up(x5)
		x_t = self.tungsten_decoder_up1(x_t, x4)
		x_t = self.tungsten_decoder_up2(x_t, x3)
		x_t = self.tungsten_decoder_up3(x_t, x2)
		t = self.tungsten_decoder_out(x_t, x1)
		x_s = self.shade_decoder_bridge_up(x5)
		x_s = self.shade_decoder_up1(x_s, x4)
		x_s = self.shade_decoder_up2(x_s, x3)
		x_s = self.shade_decoder_up3(x_s, x2)
		s = self.shade_decoder_out(x_s, x1)
		return torch.cat((awb, t, s), dim=1)
	
class deepWBnet(nn.Module):
	def __init__(self):
		super(deepWBnet, self).__init__()
		self.n_channels = 3
		self.encoder_inc = DoubleConvBlock(self.n_channels, 24)
		self.encoder_down1 = DownBlock(24, 48)
		self.encoder_down2 = DownBlock(48, 96)
		self.encoder_down3 = DownBlock(96, 192)
		self.encoder_bridge_down = BridgeDown(192, 384)
		self.decoder_bridge_up = BridgeUP(384, 192)
		self.decoder_up1 = UpBlock(192, 96)
		self.decoder_up2 = UpBlock(96, 48)
		self.decoder_up3 = UpBlock(48, 24)
		self.decoder_out = OutputBlock(24, self.n_channels)

	def forward(self, x):
		x1 = self.encoder_inc(x)
		x2 = self.encoder_down1(x1)
		x3 = self.encoder_down2(x2)
		x4 = self.encoder_down3(x3)
		x5 = self.encoder_bridge_down(x4)
		x = self.decoder_bridge_up(x5)
		x = self.decoder_up1(x, x4)
		x = self.decoder_up2(x, x3)
		x = self.decoder_up3(x, x2)
		out = self.decoder_out(x, x1)
		return out