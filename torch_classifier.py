import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import KFold
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler

from dataset import DatasetFromDirectory
from train import get_required_transforms, poly_lr_scheduler
from utils import AverageMeter, initialize_logger
from config import PATCH_SIZE, BANDS, TEST_DATASETS, TEST_ROOT_DATASET_DIR, CLASSIFIER_MODEL_NAME, DATASET_NAME, batch_size, classicication_run_title, predef_input_transform, predef_label_transform

class TorchClassifier(nn.Module):
	def __init__(self, fine_tune=True, in_channels=len(BANDS), num_classes=len(TEST_DATASETS)):
		super(TorchClassifier, self).__init__()
		self.model = EfficientNet.from_pretrained(CLASSIFIER_MODEL_NAME, advprop=True, in_channels=in_channels)
		self.linear = nn.Linear(in_features=1000, out_features=256, bias=True)
		self.relu = nn.ReLU()
		self.fc = nn.Linear(in_features=256, out_features=num_classes)

		if fine_tune:
			# print([name for name, param in self.model.named_modules()])
			for params in self.model.parameters():
				params.requires_grad = False
			for name, module in self.model.named_modules():
				if  name == '_blocks.31' or \
					name == '_fc':
					# name == '_blocks.53' or \
					# name == '_blocks.54' or \
				# if name in ['_conv_head', '_conv_head.static_padding', '_bn1', '_avg_pooling', '_dropout', '_fc', '_swish']:
					for param in module.parameters():
						param.requires_grad = True

	def forward(self, x):
		x = self.model(x)
		x = x.view(x.size(0), -1)
		x = self.relu(self.linear(x))
		# x = self.fc(x)
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

def get_loaders(input_transform, label_transform, trainset_size=0.7):
	dataset = DatasetFromDirectory(root=TEST_ROOT_DATASET_DIR,
								   dataset_name=DATASET_NAME,
								   task="classification",
								   patch_size=PATCH_SIZE,
								   lazy_read=True,
								   shuffle=True,
								   rgbn_from_cube=False,
								   use_all_bands=True,
								   product_pairing=False,
								   train_with_patches=True,
								   positive_only=True,
								   verbose=False,
								   augment_factor=8,
								   transform=(input_transform, label_transform))

	train_data, valid_data = random_split(dataset, [int(trainset_size*len(dataset)), len(dataset) - int(len(dataset)*trainset_size)])

	train_data_loader = DataLoader(dataset=train_data,
								   num_workers=1,
								   batch_size=batch_size,
								   shuffle=True,
								   pin_memory=True)

	valid_data_loader = DataLoader(dataset=valid_data,
								   num_workers=1,
								   batch_size=4,
								   shuffle=False,
								   pin_memory=True)

	return train_data_loader, valid_data_loader

init_lr = 0.00005

def main():
	logger = initialize_logger(filename="classification.log")
	history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
	log_string = "Epoch [%3d], Time: %.9f, Learning Rate: %.9f, Train Loss: %.9f, Train Accuracy: %.2f%%, Validation Loss: %.9f, Validation Accuracy: %.2f%%"

	# input_transform, label_transform = get_required_transforms(task="classification")

	print("\n" + classicication_run_title)
	logger.info(classicication_run_title)

	train_data_loader, valid_data_loader = get_loaders(predef_input_transform, predef_label_transform)

	model = TorchClassifier(fine_tune=True, in_channels=len(BANDS), num_classes=len(TEST_DATASETS))
	model = model.cuda()

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, amsgrad=True, betas=(0.9, 0.999), weight_decay=1e-5)

	iteration = 0
	for epoch in range(1, 100):
		start_time = time.time()
		train_loss, train_acc, iteration, lr = train(train_data_loader, model, criterion, iteration, optimizer)
		valid_loss, valid_acc = validate(valid_data_loader, model, criterion)

		log_string_filled = log_string % (epoch, time.time() - start_time, lr, train_loss, train_acc, valid_loss, valid_acc)

		print(log_string_filled)
		logger.info(log_string_filled)

		history["train_loss"].append(train_loss)
		history["train_acc"].append(train_acc)
		history["val_loss"].append(valid_loss)
		history["val_acc"].append(valid_acc)

	plt.plot(history["train_loss"])
	plt.plot(history["val_loss"])
	plt.title("Model Loss")
	plt.ylabel("Loss")
	plt.xlabel("Epoch")
	plt.legend(["Train", "Validation"], loc="upper left")
	plt.savefig(os.path.join(".", "inference", "losses.png"))
	plt.show()

	plt.plot(history["train_acc"])
	plt.plot(history["val_acc"])
	plt.title("Model Accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("Epoch")
	plt.legend(["Train", "Validation"], loc="upper left")
	plt.savefig(os.path.join(".", "inference", "accuracy.png"))
	plt.show()

def train(train_data_loader, model, criterion, iteration, optimizer):
	""" Trains the model on the dataloader provided """
	model.train()
	losses = AverageMeter()
	train_running_correct = 0

	for _, hypercubes, labels in tqdm(train_data_loader, desc="Train", total=len(train_data_loader)):
		# print(torch.min(images), torch.max(images), torch.min(labels), torch.max(labels))
		hypercubes = hypercubes.cuda()
		labels = labels.cuda()

		hypercubes = Variable(hypercubes)
		labels = Variable(labels)
		lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=968000, power=0.9)
		iteration = iteration + 1

		# Forward + Backward + Optimize
		optimizer.zero_grad()
		output = model(hypercubes)

		loss = criterion(output, labels)
		_, preds = torch.max(output.data, 1)
		train_running_correct += (preds == labels).sum().item()
		loss.backward()
		optimizer.step()
		losses.update(loss.item())

	epoch_acc = 100. * (train_running_correct / len(train_data_loader.dataset))
	return losses.avg, epoch_acc, iteration, lr

def validate(val_data_loader, model, criterion):
	""" Validates the model on the dataloader provided """
	model.eval()
	losses = AverageMeter()
	valid_running_correct = 0

	for  _, hypercubes, labels in tqdm(val_data_loader, desc="Valid", total=len(val_data_loader)):
		hypercubes = hypercubes.cuda()
		labels = labels.cuda()

		with torch.no_grad():
			hypercubes = Variable(hypercubes)
			labels = Variable(labels)

			output = model(hypercubes)

		loss = criterion(output, labels)
		losses.update(loss.item())
		_, preds = torch.max(output.data, 1)
		valid_running_correct += (preds == labels).sum().item()

	epoch_acc = 100. * (valid_running_correct / len(val_data_loader.dataset))
	return losses.avg, epoch_acc

if __name__ == "__main__":
	main()