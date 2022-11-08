import os
import time
import json
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

from models.classifier import TorchClassifier, SeparateClassifier

from dataset import DatasetFromDirectory
from utils import AverageMeter, initialize_logger, save_checkpoint, poly_lr_scheduler, get_required_transforms, get_dataloaders
from config import PATCH_SIZE, BANDS, TEST_DATASETS, TEST_ROOT_DATASET_DIR, CLASSIFIER_MODEL_NAME, DATASET_NAME, batch_size, classicication_run_title, predef_input_transform, predef_label_transform, init_directories

init_lr = 0.00005

activations = {}
def get_activation(name):
	def hook(model, input, output):
		activations[name] = output.detach()
	return hook

class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

def main():
	logger = initialize_logger(filename="classification.log")
	history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
	log_string = "Epoch [%3d], Time: %.9f, Learning Rate: %.9f, Train Loss: %.9f, Train Accuracy: %.2f%%, Validation Loss: %.9f, Validation Accuracy: %.2f%%"

	# input_transform, label_transform = get_required_transforms(task="classification")

	print("\n" + classicication_run_title)
	logger.info(classicication_run_title)

	train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(predef_input_transform, predef_label_transform)

	model = TorchClassifier(fine_tune=True, in_channels=len(BANDS), num_classes=len(TEST_DATASETS))
	model.bottleneck.register_forward_hook(get_activation("bottleneck"))
	model = model.cuda()

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, amsgrad=True, betas=(0.9, 0.999), weight_decay=1e-5)

	iteration = 0
	for epoch in range(1, 25):
		start_time = time.time()
		train_loss, train_acc, iteration, lr = train(train_data_loader, model, criterion, iteration, optimizer)
		val_loss, val_acc = validate(valid_data_loader, model, criterion)
		save_checkpoint(epoch, iteration, model, optimizer, val_loss, val_acc, task="classification")

		log_string_filled = log_string % (epoch, time.time() - start_time, lr, train_loss, train_acc, val_loss, val_acc)

		print(log_string_filled)
		logger.info(log_string_filled)

		history["train_loss"].append(train_loss)
		history["train_acc"].append(train_acc)
		history["val_loss"].append(val_loss)
		history["val_acc"].append(val_acc)

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
	test_loss, test_acc, json_data = test(test_data_loader, model, criterion)
	print("Test Loss: %.9f, Test Accuracy: %.2f%%" % (test_loss, test_acc))

	jsonFile = open(os.path.join("inference", "weights_HS.json"), "w")
	jsonFile.write(json.dumps(json_data, indent=4))
	jsonFile.close()

def train(train_data_loader, model, criterion, iteration, optimizer):
	""" Trains the model on the dataloader provided """
	model.train()
	losses = AverageMeter()
	train_running_correct = 0

	for rgbn, hypercubes, labels in tqdm(train_data_loader, desc="Train", total=len(train_data_loader)):
		# rgbn = rgbn[:, :3, :, :]
		# rgbn = rgbn.cuda()
		hypercubes = hypercubes.cuda()
		labels = labels.cuda()

		# rgbn = Variable(rgbn)
		hypercubes = Variable(hypercubes)
		labels = Variable(labels)
		lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=968000, power=0.9)
		iteration = iteration + 1

		# Forward + Backward + Optimize
		optimizer.zero_grad()
		# output = model(rgbn)
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
	correct_examples = 0

	for rgbn, hypercubes, labels in tqdm(val_data_loader, desc="Valid", total=len(val_data_loader)):
		# rgbn = rgbn[:, :3, :, :]
		# rgbn = rgbn.cuda()
		hypercubes = hypercubes.cuda()
		labels = labels.cuda()

		with torch.no_grad():
			# rgbn = Variable(rgbn)
			hypercubes = Variable(hypercubes)
			labels = Variable(labels)

			# output = model(rgbn)
			output = model(hypercubes)

		loss = criterion(output, labels)
		losses.update(loss.item())
		_, preds = torch.max(output.data, 1)
		correct_examples += (preds == labels).sum().item()

	epoch_acc = 100. * (correct_examples / len(val_data_loader.dataset))
	return losses.avg, epoch_acc

def test(test_data_loader, model, criterion):
	""" Tests the model on the dataloader provided """
	model.eval()
	losses = AverageMeter()
	correct_examples = 0
	y_pred, y_true = [], []
	json_data = []

	for rgbn, hypercubes, labels in tqdm(test_data_loader, desc="Test", total=len(test_data_loader)):
		y_true.extend(labels.data.numpy())
		# rgbn = rgbn[:, :3, :, :]
		# rgbn = rgbn.cuda()
		hypercubes = hypercubes.cuda()
		labels = labels.cuda()

		with torch.no_grad():
			# rgbn = Variable(rgbn)
			hypercubes = Variable(hypercubes)
			labels = Variable(labels)

			# output = model(rgbn)
			output = model(hypercubes)
		json_data.append({"output": activations["bottleneck"].cpu().numpy().tolist(), "label": int(labels.cpu().numpy()[0])})
		loss = criterion(output, labels)
		losses.update(loss.item())
		_, preds = torch.max(output.data, 1)
		y_pred.extend(preds.data.cpu().numpy())
		correct_examples += (preds == labels).sum().item()

	accuracy = 100. * (correct_examples / len(test_data_loader.dataset))

	print("Weights Shape:", activations["bottleneck"].cpu().numpy().shape)

	confusion_mat = confusion_matrix(y_true, y_pred)
	sns.heatmap(confusion_mat/np.sum(confusion_mat), annot=True, fmt=".2%")
	plt.savefig(os.path.join("inference", "confusion_matrix.png"))
	plt.show()

	return losses.avg, accuracy, json_data

if __name__ == "__main__":
	init_directories()
	main()