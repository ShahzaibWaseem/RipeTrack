import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BANDS, LABELS_DICT, SUB_LABELS_DICT

class FeatureExtractionBlock(nn.Module):
	def __init__(self, input_channels):
		super().__init__()
		self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(input_channels)
		self.relu = nn.LeakyReLU(inplace=True)
		self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(input_channels)

	def forward(self, x):
		# residual = x
		x = self.bn1(self.conv1(x))
		x = self.relu(x)
		x = self.bn2(self.conv2(x))
		return x

class ModelWithAttention(nn.Module):
	def __init__(self, input_channels=len(BANDS), num_classes=len(LABELS_DICT), num_subclasses=len(SUB_LABELS_DICT)):
		super().__init__()
		self.ssattn = SSAttention(input_channels)
		self.relu = nn.LeakyReLU(inplace=True)
		self.bottleneck = nn.Linear(in_features=296208, out_features=256)
		self.dropout = nn.Dropout(p=0.25)
		self.fc_class = nn.Linear(in_features=256, out_features=num_classes)
		self.fc_subclass = nn.Linear(in_features=256, out_features=num_subclasses)
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if(isinstance(module, nn.Conv2d)):
			nn.init.xavier_uniform_(module.weight)
		elif isinstance(module, nn.BatchNorm2d):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)

	def forward(self, x):
		x = self.ssattn(x)
		x = x.view(x.size(0), -1)
		x = self.bottleneck(x)
		x = self.relu(x)
		x = self.dropout(x)
		class_out = self.fc_class(x)
		subclass_out = self.fc_subclass(x)
		return class_out, subclass_out

class SSAttention(nn.Module):
	def __init__(self, channels_in):
		super(SSAttention, self).__init__()
		self.feature_extractor = FeatureExtractionBlock(channels_in)
		self.cbam = CBAM(channels_in, 2, 3)
		self.relu = nn.LeakyReLU(inplace=True)
		self.bn1 = nn.BatchNorm2d(channels_in)
		self.conv1 = nn.Conv2d(channels_in, channels_in, kernel_size=1, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(channels_in)

	def forward(self, x):
		x = self.feature_extractor(x)
		x = self.cbam(x)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.conv1(x)
		x = self.relu(x)
		x = self.bn2(x)
		return x

class CBAM(nn.Module):
	def __init__(self, n_channels_in, reduction_ratio, kernel_size):
		super(CBAM, self).__init__()
		self.n_channels_in = n_channels_in
		self.reduction_ratio = reduction_ratio
		self.kernel_size = kernel_size

		self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
		self.spatial_attention = SpatialAttention(kernel_size)

	def forward(self, f):
		chan_att = self.channel_attention(f)
		fp = chan_att * f
		spat_att = self.spatial_attention(fp)
		fpp = spat_att * fp
		return fpp

class SpatialAttention(nn.Module):
	def __init__(self, kernel_size):
		super(SpatialAttention, self).__init__()
		self.kernel_size = kernel_size

		assert kernel_size % 2 == 1, "Odd kernel size required"
		self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=int((kernel_size-1)/2))

	def forward(self, x):
		max_pool = self.agg_channel(x, "max")
		avg_pool = self.agg_channel(x, "avg")
		pool = torch.cat([max_pool, avg_pool], dim = 1)
		conv = self.conv(pool)
		conv = conv.repeat(1,x.size()[1],1,1)
		att = torch.sigmoid(conv)
		return att

	def agg_channel(self, x, pool = "max"):
		b,c,h,w = x.size()
		x = x.view(b, c, h*w)
		x = x.permute(0, 2, 1)
		if pool == "max":
			x = F.max_pool1d(x, c)
		elif pool == "avg":
			x = F.avg_pool1d(x, c)
		x = x.permute(0, 2, 1)
		x = x.view(b, 1, h, w)
		return x

class ChannelAttention(nn.Module):
	def __init__(self, n_channels_in, reduction_ratio):
		super(ChannelAttention, self).__init__()
		self.n_channels_in = n_channels_in
		self.reduction_ratio = reduction_ratio
		self.middle_layer_size = int(self.n_channels_in/float(self.reduction_ratio))

		self.bottleneck = nn.Sequential(
			nn.Linear(self.n_channels_in, self.middle_layer_size),
			nn.ReLU(),
			nn.Linear(self.middle_layer_size, self.n_channels_in)
		)

	def forward(self, x):
		kernel = (x.size()[2], x.size()[3])
		avg_pool = F.avg_pool2d(x, kernel)
		max_pool = F.max_pool2d(x, kernel)

		avg_pool = avg_pool.view(avg_pool.size()[0], -1)
		max_pool = max_pool.view(max_pool.size()[0], -1)

		avg_pool_bck = self.bottleneck(avg_pool)
		max_pool_bck = self.bottleneck(max_pool)

		pool_sum = avg_pool_bck + max_pool_bck

		sig_pool = torch.sigmoid(pool_sum)
		sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

		out = sig_pool.repeat(1, 1, kernel[0], kernel[1])
		return out