import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from config import BANDS, TEST_DATASETS, CLASSIFIER_MODEL_NAME, PATCH_SIZE

class TorchClassifier(nn.Module):
	def __init__(self, fine_tune=True, in_channels=len(BANDS), num_classes=len(TEST_DATASETS)):
		super(TorchClassifier, self).__init__()
		self.model = EfficientNet.from_pretrained(CLASSIFIER_MODEL_NAME, advprop=True, in_channels=in_channels)
		self.bottleneck = nn.Linear(in_features=1000, out_features=256, bias=True)
		self.relu = nn.ReLU()
		self.fc = nn.Linear(in_features=256, out_features=num_classes)

		if fine_tune:
			# print([name for name, param in self.model.named_modules()])
			for params in self.model.parameters():
				params.requires_grad = False
			for name, module in self.model.named_modules():
				if  name == "_blocks.31" or \
					name == "_fc":
					# name == "_blocks.53" or \
					# name == "_blocks.54" or \
				# if name in ["_conv_head", "_conv_head.static_padding", "_bn1", "_avg_pooling", "_dropout", "_fc", "_swish"]:
					for param in module.parameters():
						param.requires_grad = True

	def forward(self, x):
		x = self.model(x)
		x = x.view(x.size(0), -1)
		x = self.relu(self.bottleneck(x))
		x = self.fc(x)
		return x

class SeparateClassifiers(nn.Module):
	def __init__(self, fine_tune=True, in_channels=len(BANDS), num_classes=len(TEST_DATASETS)):
		super().__init__()
		self.vis_module = TorchClassifier(fine_tune=fine_tune, in_channels=3, num_classes=num_classes)
		self.nir_module = TorchClassifier(fine_tune=fine_tune, in_channels=in_channels-3, num_classes=num_classes)
		self.relu = nn.ReLU()
		self.fc = nn.Linear(in_features=2*256, out_features=num_classes)

	def forward(self, x):
		rgb_x = x[:, :3, :, :]
		nir_x = x[:, 3:, :, :]
		rgb_x = self.vis_module(rgb_x)
		nir_x = self.nir_module(nir_x)
		x = torch.cat((rgb_x, nir_x), dim=1)
		x = self.relu(self.fc(x))
		return x

class FeatureExtractionBlock(nn.Module):
	def __init__(self, input_channels):
		super().__init__()
		intermediate_channels = 16
		self.conv1 = nn.Conv2d(input_channels, intermediate_channels, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(intermediate_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(intermediate_channels)

	def forward(self, x):
		# residual = x
		x = self.bn1(self.conv1(x))
		x = self.relu(x)
		x = self.bn2(self.conv2(x))
		# x = torch.mul(x, 0.1)
		# x = torch.add(x, residual)
		return x

class MultiHeadClassification(nn.Module):
	def __init__(self, input_channels=3, groups=3, num_classes=2):
		super().__init__()
		bands_head1, bands_head2 = BANDS[:16], BANDS[-17:]
		self.feature_ex_block1 = nn.Sequential(FeatureExtractionBlock(len(bands_head1)), nn.MaxPool2d(2, 2), FeatureExtractionBlock(16), FeatureExtractionBlock(16), FeatureExtractionBlock(16))
		self.feature_ex_block2 = nn.Sequential(FeatureExtractionBlock(len(bands_head2)), nn.MaxPool2d(2, 2), FeatureExtractionBlock(16), FeatureExtractionBlock(16), FeatureExtractionBlock(16))
		# self.fusion_layer = nn.Conv2d(16+16, 32, kernel_size=3, stride=1, padding=1)
		# self.bn = nn.BatchNorm2d(32)
		self.relu = nn.ReLU(inplace=True)
		self.bottleneck = nn.Linear(in_features=32*PATCH_SIZE*PATCH_SIZE//4, out_features=256, bias=True)
		self.dropout = nn.Dropout(p=0.5)
		self.fc = nn.Linear(in_features=256, out_features=num_classes)

	def forward(self, x):
		x1 = x[:, :16, :, :]
		x2 = x[:, -17:, :, :]
		x1 = self.feature_ex_block1(x1)
		x2 = self.feature_ex_block2(x2)
		x = torch.cat((x1, x2), dim=1)
		# x = self.relu(self.bn(self.fusion_layer(x)))
		x = x.view(x.size(0), -1)
		x = self.relu(self.bottleneck(x))
		x = self.dropout(x)
		x = self.fc(x)
		return x