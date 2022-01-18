import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels):
	return nn.Conv2d(in_channels, out_channels, kernel_size=3,
					 stride=1, padding=1, bias=True)

class resblock(nn.Module):
	def __init__(self):
		super(resblock, self).__init__()
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