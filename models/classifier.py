import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from config import BANDS, TEST_DATASETS, CLASSIFIER_MODEL_NAME

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