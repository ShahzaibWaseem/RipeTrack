import torch
import torch.nn as nn
from math import sqrt

from models.gnn import GraphNet

def conv3x3(in_channels, out_channels):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3,
					 stride=1, padding=1, bias=True)

class conv_bn_relu_res_block(nn.Module):
	def __init__(self):
		super(conv_bn_relu_res_block, self).__init__()
		self.conv1 = conv3x3(64, 64)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(64, 64)
		self.bn2 = nn.BatchNorm2d(64)

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = torch.mul(out,0.1)
		out = torch.add(out,residual)
		return out

class resblock(nn.Module):
	def __init__(self, block, block_num, input_channel, output_channel, fusion="concat"):
		super(resblock, self).__init__()

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

		# self.downsample_1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
		# self.downsample_prelu_1 = nn.PReLU()
		# self.downsample_2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
		# self.downsample_prelu_2 = nn.PReLU()
		# self.downsample_3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
		# self.downsample_prelu_3 = nn.PReLU()

		# self.upsample_1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
		# self.upsample_prelu_1 = nn.PReLU()
		# self.upsample_2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
		# self.upsample_prelu_2 = nn.PReLU()
		# self.upsample_3 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
		# self.upsample_prelu_3 = nn.PReLU()

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

	def split_tensor(self, image, split_size=64):
		torch.cuda.empty_cache()
		size = image.size(2)
		output = torch.zeros(image.shape, dtype=torch.float32).cuda()
		for split in range(size//split_size):
			start, end = split*split_size, (split+1)*split_size
			output[:, :, start:end, start:end] = self.gnn_block(image[:, :, start:end, start:end])
		return output

	def make_layer(self, block, num_layers):
		layers = []
		for i in range(num_layers):
			layers.append(block())
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.input_conv(x)
		out = self.input_relu(out)

		gnn_out = self.downsampling_block(out)

		if (x.size(2) > 64):
			gnn_out = self.split_tensor(gnn_out, split_size=64)
		else:
			gnn_out = self.gnn_block(gnn_out)

		gnn_out = self.upsampling_block(gnn_out)

		residual = out
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