import os
import time
import json
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from models.classifier import TorchClassifier, SeparateClassifiers, MultiHeadClassification, ModelWithAttention

from dataset import get_dataloaders_classification
from utils import AverageMeter, initialize_logger, save_checkpoint, poly_lr_scheduler, get_best_checkpoint
from config import VISUALIZATION_DIR_NAME, MODEL_PATH, LOGS_PATH, BANDS, LABELS_DICT, run_pretrained, classicication_run_title, run_pretrained, end_epoch, batch_size, create_directory

init_lr = 0.0005
y_pred, y_true = [], []

parser = argparse.ArgumentParser()
parser.add_argument("--disable_tqdm", default=False, required=False, type=bool, help="Disable tqdm progress bar")
args = parser.parse_args()
disable_tqdm = args.disable_tqdm

def get_label_weights(train_data_loader, val_data_loader):
	class_labels = []
	for _, labels, _ in train_data_loader:
		class_labels.extend(labels.numpy().reshape(-1,))
	for _, labels, _ in val_data_loader:
		class_labels.extend(labels.numpy().reshape(-1,))
	class_labels = np.asarray(class_labels)
	class_labels = class_labels.reshape((-1,))
	unique_y = np.unique(class_labels)
	class_weights = compute_class_weight("balanced", unique_y, class_labels)
	return class_weights

def test_model_only():
	test_data_loader, _ = get_dataloaders_classification(trainset_size=1.0)
	model = ModelWithAttention(input_channels=len(BANDS), num_classes=len(LABELS_DICT))
	model = model.cuda()
	criterion = torch.nn.CrossEntropyLoss(reduction="mean")
	best_checkpoint_file, epoch, iter, state_dict, optimizer, val_loss, val_acc = get_best_checkpoint(task="classification")
	model.load_state_dict(state_dict)
	test_loss, test_acc = test(test_data_loader, model, criterion)

def main():
	logger = initialize_logger(filename="classification.log")
	history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
	log_string = "Epoch [%3d], Iter[%5d], Time: %.9f, Train Loss: %.9f, Train Accuracy: %.2f%%, Validation Loss: %.9f, Validation Accuracy: %.2f%%"

	# input_transform, label_transform = get_required_transforms(task="classification")

	print("\n" + classicication_run_title)
	logger.info(classicication_run_title)
	trainset_size = 0.85

	# train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(predef_input_transform, predef_label_transform, task="classification")
	train_data_loader, valid_data_loader = get_dataloaders_classification(trainset_size)
	whole_dataset_size = len(train_data_loader.dataset) + len(valid_data_loader.dataset)
	class_weights = get_label_weights(train_data_loader, valid_data_loader)
	print("Class Weights Loss Function: {}".format(class_weights))
	class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()

	# model = TorchClassifier(fine_tune=True, in_channels=len(BANDS), num_classes=len(LABELS_DICT))
	model = ModelWithAttention(input_channels=len(BANDS), num_classes=len(LABELS_DICT))
	# model.bottleneck.register_forward_hook(get_activation("bottleneck"))
	model = model.cuda()

	criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, amsgrad=True, betas=(0.9, 0.999), weight_decay=1e-6)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5, verbose=True)

	epoch, iteration, best_epoch, best_val_loss, best_val_acc = 0, 0, 0, 0, 0

	if run_pretrained:
		epoch, iteration, state_dict, optimizer, val_loss, val_acc = get_best_checkpoint(task="classification")
		model.load_state_dict(state_dict)

	for epoch in range(1, end_epoch):
		start_time = time.time()
		train_loss, train_acc, iteration = train(train_data_loader, model, criterion, iteration, optimizer)
		val_loss, val_acc = validate(valid_data_loader, model, criterion)
		if (100 - best_val_loss + best_val_acc) < (100 - val_loss + val_acc):
			best_val_acc = val_acc
			best_val_loss = val_loss
			best_epoch = epoch
			best_model = model
			best_optimizer = optimizer
			iteration_passed = iteration
		if epoch % 10 == 0:
			save_checkpoint(best_epoch, iteration_passed, best_model, best_optimizer, best_val_loss, best_val_acc, bands=BANDS, task="classification")
		if epoch % 100 == 0:
			test_loss, test_acc = test(valid_data_loader, model, criterion)
		# scheduler.step(val_loss)

		log_string_filled = log_string % (epoch, iteration, time.time() - start_time, train_loss, train_acc, val_loss, val_acc)

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
	plt.savefig(os.path.join(LOGS_PATH, "losses.png"))
	plt.show()

	plt.plot(history["train_acc"])
	plt.plot(history["val_acc"])
	plt.title("Model Accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("Epoch")
	plt.legend(["Train", "Validation"], loc="upper left")
	plt.savefig(os.path.join(LOGS_PATH, "accuracy.png"))
	plt.show()
	# test_loss, test_acc, json_data = test(test_data_loader, model, criterion)
	# print("Test Loss: %.9f, Test Accuracy: %.2f%%" % (test_loss, test_acc))

	# jsonFile = open(os.path.join("inference", "weights_HS.json"), "w")
	# jsonFile.write(json.dumps(json_data, indent=4, cls=NumpyEncoder))
	# jsonFile.close()

def train(train_data_loader, model, criterion, iteration, optimizer):
	""" Trains the model on the dataloader provided """
	model.train()
	losses = AverageMeter()
	train_running_correct = 0

	for hypercubes, labels, _ in tqdm(train_data_loader, desc="Train", total=len(train_data_loader), disable=disable_tqdm):
		hypercubes = hypercubes.cuda()
		labels = labels.cuda()

		hypercubes = Variable(hypercubes)
		labels = Variable(labels)
		# lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=max_iter, power=0.75)
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
	return losses.avg, epoch_acc, iteration

def validate(val_data_loader, model, criterion):
	""" Validates the model on the dataloader provided """
	model.eval()
	losses = AverageMeter()
	correct_examples = 0

	for hypercubes, labels, _ in tqdm(val_data_loader, desc="Valid", total=len(val_data_loader), disable=disable_tqdm):
		hypercubes = hypercubes.cuda()
		labels = labels.cuda()

		with torch.no_grad():
			hypercubes = Variable(hypercubes)
			labels = Variable(labels)

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
	y_pred, y_true, fruit_labels = [], [], []

	for hypercubes, labels, fruits in tqdm(test_data_loader, desc="Test", total=len(test_data_loader)):
		y_true.extend(labels.data.numpy())
		hypercubes = hypercubes.cuda()
		labels = labels.cuda()

		with torch.no_grad():
			hypercubes = Variable(hypercubes)
			labels = Variable(labels)

			output = model(hypercubes)
		loss = criterion(output, labels)
		losses.update(loss.item())
		_, preds = torch.max(output.data, 1)
		y_pred.extend(preds.data.cpu().numpy())
		fruit_labels.extend(fruits)
		correct_examples += (preds == labels).sum().item()

	pear_bosc_indices = find_indices(fruit_labels, "Pear Bosc")
	pear_williams_indices = find_indices(fruit_labels, "Pear Williams")
	avo_empire_indices = find_indices(fruit_labels, "Avocado Emp")
	avo_organic_indices = find_indices(fruit_labels, "Avocado Organic")
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	accuracy = 100. * (correct_examples / len(test_data_loader.dataset))
	classification_evaluate(y_true, y_pred, "all")
	classification_evaluate(y_true[pear_bosc_indices], y_pred[pear_bosc_indices], "pear_bosc")
	classification_evaluate(y_true[pear_williams_indices], y_pred[pear_williams_indices], "pear_williams")
	classification_evaluate(y_true[avo_empire_indices], y_pred[avo_empire_indices], "avocado_emp")
	classification_evaluate(y_true[avo_organic_indices], y_pred[avo_organic_indices], "avocado_organic")

	return losses.avg, accuracy

def find_indices(list, fruit):
	return [i for i, x in enumerate(list) if x == fruit]

def classification_evaluate(y_true, y_pred, title):
	confusion_mat = confusion_matrix(y_true, y_pred)
	df_confusion_mat = pd.DataFrame(confusion_mat / np.sum(confusion_mat, axis=1)[:, None], index = [key for key, value in LABELS_DICT.items()], columns = [key for key, value in LABELS_DICT.items()])
	plt.figure()
	sns.heatmap(df_confusion_mat, annot=True, fmt=".2%")
	print("Title: {}, Accuracy: {}".format(title, accuracy_score(y_true, y_pred)))
	print(classification_report(y_true, y_pred, target_names=[key for key, value in LABELS_DICT.items()]))
	print(df_confusion_mat)
	plt.savefig(os.path.join(VISUALIZATION_DIR_NAME, "confusion_matrix_{}.png".format(title)))
	plt.show()

if __name__ == "__main__":
	create_directory(os.path.join(VISUALIZATION_DIR_NAME))
	create_directory(os.path.join(MODEL_PATH, "classification"))
	main()
	# test_model_only()