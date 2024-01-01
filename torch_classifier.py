import os
import time
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from models.classifier import ModelWithAttention

from dataset import get_dataloaders_classification
from utils import AverageMeter, initialize_logger, save_checkpoint, get_best_checkpoint
from config import BANDS, VISUALIZATION_DIR_NAME, MODEL_PATH, LABELS_DICT, SUB_LABELS_DICT, TIME_LEFT_DICT, classicication_run_title, end_epoch, run_pretrained, create_directory

init_lr = 0.0005
y_pred, y_true = [], []

parser = argparse.ArgumentParser()
parser.add_argument("--disable_tqdm", default=False, required=False, type=bool, help="Disable tqdm progress bar")
args = parser.parse_args()
disable_tqdm = args.disable_tqdm

def get_label_weights(train_data_loader, val_data_loader):
	class_labels = []
	for _, labels, sublabels, _ in train_data_loader:
		class_labels.extend(labels.numpy().reshape(-1,))
	for _, labels, sublabels, _ in val_data_loader:
		class_labels.extend(labels.numpy().reshape(-1,))
	class_labels = np.asarray(class_labels)
	class_labels = class_labels.reshape((-1,))
	unique_y = np.unique(class_labels)
	class_weights = compute_class_weight(class_weight="balanced", classes=unique_y, y=class_labels)
	return class_weights

def test_model_only():
	best_checkpoint_file, epoch, iter, state_dict, optimizer, val_loss, (val_acc_labels, val_acc_sublabels) = get_best_checkpoint(task="classification")
	test_data_loader, _ = get_dataloaders_classification(trainset_size=1.0)
	model = ModelWithAttention(input_channels=len(BANDS), num_classes=len(LABELS_DICT), num_subclasses=len(TIME_LEFT_DICT))
	model = model.cuda()
	criterion = (torch.nn.CrossEntropyLoss(reduction="mean"), torch.nn.CrossEntropyLoss(reduction="mean"))
	model.load_state_dict(state_dict)
	test_loss, test_acc = test(test_data_loader, model, criterion)

def main():
	logger = initialize_logger(filename="classification.log")
	history = {"train_loss": [], "train_loss_labels": [], "train_loss_sublabels": [], "train_acc_labels": [], "train_acc_sublabels": [], "val_loss": [], "val_loss_labels": [], "val_loss_sublabels": [], "val_acc_labels": [], "val_acc_sublabels": []}
	log_string = "Epoch [%3d], Iter[%5d], Time: %.2f, Train Loss: %.8f (%.8f, %.8f), Train Accuracy: %.2f%%, %.2f%%, Validation Loss: %.8f (%.8f, %.8f), Validation Accuracy: %.2f%%, %.2f%%"

	# input_transform, label_transform = get_required_transforms(task="classification")

	print("\n" + classicication_run_title)
	logger.info(classicication_run_title)
	trainset_size = 0.85

	# train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(predef_input_transform, predef_label_transform, task="classification")
	train_data_loader, valid_data_loader = get_dataloaders_classification(trainset_size)
	torch.save(valid_data_loader, os.path.join(MODEL_PATH, "valid_data_loader.pt"))
	whole_dataset_size = len(train_data_loader.dataset) + len(valid_data_loader.dataset)
	class_weights = get_label_weights(train_data_loader, valid_data_loader)
	print("Class Weights Loss Function: {}".format(class_weights))
	class_weights = torch.tensor(class_weights, dtype=torch.float32).cuda()

	# model = TorchClassifier(fine_tune=True, in_channels=len(BANDS), num_classes=len(LABELS_DICT))
	model = ModelWithAttention(input_channels=len(BANDS), num_classes=len(LABELS_DICT), num_subclasses=len(TIME_LEFT_DICT))
	# model.bottleneck.register_forward_hook(get_activation("bottleneck"))
	model = model.cuda()

	criterion_class = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
	criterion_subclass = torch.nn.CrossEntropyLoss()
	criterion = (criterion_class, criterion_subclass)
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, amsgrad=True, betas=(0.9, 0.999), weight_decay=1e-6)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5, verbose=True)

	epoch, iteration, best_epoch, best_val_loss, best_val_acc_labels, best_val_acc_sublabels = 0, 0, 0, 0, 0, 0

	if run_pretrained:
		epoch, iteration, state_dict, optimizer, val_loss, (val_acc_labels, val_acc_sublabels) = get_best_checkpoint(task="classification")
		model.load_state_dict(state_dict)

	for epoch in range(1, end_epoch):
		start_time = time.time()
		(train_loss, train_loss_labels, train_loss_sublabels), (train_acc_labels, train_acc_sublabels), iteration = train(train_data_loader, model, criterion, iteration, optimizer)
		(val_loss, val_loss_labels, val_loss_sublabels), (val_acc_labels, val_acc_sublabels) = validate(valid_data_loader, model, criterion)
		if (100 - best_val_loss + best_val_acc_labels + best_val_acc_sublabels) < (100 - val_loss + val_acc_labels + val_acc_sublabels):
			best_val_loss = val_loss
			best_val_acc_labels = val_acc_labels
			best_val_acc_sublabels = val_acc_sublabels
			best_epoch = epoch
			best_model = model
			best_optimizer = optimizer
			iteration_passed = iteration
		if epoch % 30 == 0:
			if epoch <= 150:
				continue
			else:
				save_checkpoint(int(round(epoch, -1)), iteration_passed, best_model, best_optimizer, best_val_loss, best_val_acc_labels, best_val_acc_sublabels, bands=BANDS, task="classification")
		if epoch % 100 == 0:
			test_loss, test_acc = test(valid_data_loader, model, criterion)
		# scheduler.step(val_loss)

		log_string_filled = log_string % (epoch, iteration, time.time() - start_time, train_loss, train_loss_labels, train_loss_sublabels, train_acc_labels, train_acc_sublabels, val_loss, val_loss_labels, val_loss_sublabels, val_acc_labels, val_acc_sublabels)

		print(log_string_filled)
		logger.info(log_string_filled)

		history["train_loss"].append(train_loss)
		history["train_loss_labels"].append(train_loss_labels)
		history["train_loss_sublabels"].append(train_loss_sublabels)
		history["train_acc_labels"].append(train_acc_labels)
		history["train_acc_sublabels"].append(train_acc_sublabels)
		history["val_loss"].append(val_loss)
		history["val_loss_labels"].append(val_loss_labels)
		history["val_loss_sublabels"].append(val_loss_sublabels)
		history["val_acc_labels"].append(val_acc_labels)
		history["val_acc_sublabels"].append(val_acc_sublabels)

	plt.plot(history["train_loss"])
	plt.plot(history["val_loss"])
	plt.title("Model Loss")
	plt.ylabel("Loss")
	plt.xlabel("Epoch")
	plt.legend(["Train", "Validation"], loc="upper left")
	plt.savefig(os.path.join(VISUALIZATION_DIR_NAME, "losses.pdf"))
	plt.show()
	plt.close()

	plt.plot(history["train_acc_labels"], label="Train Label")
	plt.plot(history["val_acc_labels"], label="Validation Label")
	plt.plot(history["train_acc_sublabels"], label="Train Label")
	plt.plot(history["val_acc_sublabels"], label="Validation SubLabel")
	plt.title("Model Accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("Epoch")
	plt.legend(loc="upper left")
	plt.savefig(os.path.join(VISUALIZATION_DIR_NAME, "accuracy.pdf"))
	plt.show()
	plt.close()

def train(train_data_loader, model, criterion, iteration, optimizer):
	""" Trains the model on the dataloader provided """
	model.train()
	losses, losses_class, losses_subclass = AverageMeter(), AverageMeter(), AverageMeter()
	running_correct_labels, running_correct_sublabels = 0, 0
	criterion_class, criterion_subclass = criterion

	for hypercubes, labels, sublabels, _ in tqdm(train_data_loader, desc="Train", total=len(train_data_loader), disable=disable_tqdm):
		hypercubes = hypercubes.cuda()
		labels = labels.cuda()
		sublabels = sublabels.cuda()

		hypercubes = Variable(hypercubes)
		labels = Variable(labels)
		# lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=max_iter, power=0.75)
		iteration = iteration + 1

		# Forward + Backward + Optimize
		optimizer.zero_grad()
		out_labels, out_sublabels = model(hypercubes)
		_, preds_labels = torch.max(out_labels.data, 1)
		_, preds_sublabels = torch.max(out_sublabels.data, 1)

		loss_class = criterion_class(out_labels, labels)
		loss_subclass = criterion_subclass(out_sublabels, sublabels)
		# loss_penalize = # TODO: Penalty for misclassifying subclass inside a class (non-correspondance) 
		loss = loss_class + loss_subclass # + loss_penalize

		running_correct_labels += (preds_labels == labels).sum().item()
		running_correct_sublabels += (preds_sublabels == sublabels).sum().item()
		loss.backward()
		optimizer.step()
		losses.update(loss.item())
		losses_class.update(loss_class.item())
		losses_subclass.update(loss_subclass.item())

	epoch_acc_labels = 100. * (running_correct_labels / len(train_data_loader.dataset))
	epoch_acc_sublabels = 100. * (running_correct_sublabels / len(train_data_loader.dataset))
	return (losses.avg, losses_class.avg, losses_subclass.avg), (epoch_acc_labels, epoch_acc_sublabels), iteration

def validate(val_data_loader, model, criterion):
	""" Validates the model on the dataloader provided """
	model.eval()
	losses, losses_class, losses_subclass = AverageMeter(), AverageMeter(), AverageMeter()
	running_correct_labels, running_correct_sublabels = 0, 0
	criterion_class, criterion_subclass = criterion

	for hypercubes, labels, sublabels, _ in tqdm(val_data_loader, desc="Valid", total=len(val_data_loader), disable=disable_tqdm):
		hypercubes = hypercubes.cuda()
		labels = labels.cuda()
		sublabels = sublabels.cuda()

		with torch.no_grad():
			hypercubes = Variable(hypercubes)
			labels = Variable(labels)
			out_labels, out_sublabels = model(hypercubes)

		_, preds_labels = torch.max(out_labels.data, 1)
		_, preds_sublabels = torch.max(out_sublabels.data, 1)

		loss_class = criterion_class(out_labels, labels)
		loss_subclass = criterion_subclass(out_sublabels, sublabels)
		loss = loss_class + loss_subclass

		running_correct_labels += (preds_labels == labels).sum().item()
		running_correct_sublabels += (preds_sublabels == sublabels).sum().item()

		losses.update(loss.item())
		losses_class.update(loss_class.item())
		losses_subclass.update(loss_subclass.item())

	epoch_acc_labels = 100. * (running_correct_labels / len(val_data_loader.dataset))
	epoch_acc_sublabels = 100. * (running_correct_sublabels / len(val_data_loader.dataset))
	return (losses.avg, losses_class.avg, losses_subclass.avg), (epoch_acc_labels, epoch_acc_sublabels)

def test(test_data_loader, model, criterion):
	""" Tests the model on the dataloader provided """
	model.eval()
	losses, losses_class, losses_subclass = AverageMeter(), AverageMeter(), AverageMeter()
	running_correct_labels, running_correct_sublabels = 0, 0
	criterion_class, criterion_subclass = criterion
	y_pred_labels, y_true_labels, y_pred_sublabels, y_true_sublabels, fruit_labels = [], [], [], [], []

	for hypercubes, labels, sublabels, fruits in tqdm(test_data_loader, desc="Test", total=len(test_data_loader), disable=disable_tqdm):
		y_true_labels.extend(labels.data.numpy())
		y_true_sublabels.extend(sublabels.data.numpy())
		hypercubes = hypercubes.cuda()
		labels = labels.cuda()
		sublabels = sublabels.cuda()

		with torch.no_grad():
			hypercubes = Variable(hypercubes)
			labels = Variable(labels)
			out_labels, out_sublabels = model(hypercubes)

		_, preds_labels = torch.max(out_labels.data, 1)
		_, preds_sublabels = torch.max(out_sublabels.data, 1)

		loss_class = criterion_class(out_labels, labels)
		loss_subclass = criterion_subclass(out_sublabels, sublabels)
		loss = loss_class + loss_subclass

		running_correct_labels += (preds_labels == labels).sum().item()
		running_correct_sublabels += (preds_sublabels == sublabels).sum().item()

		y_pred_labels.extend(preds_labels.data.cpu().numpy())
		y_pred_sublabels.extend(preds_sublabels.data.cpu().numpy())
		fruit_labels.extend(fruits)

		losses.update(loss.item())
		losses_class.update(loss_class.item())
		losses_subclass.update(loss_subclass.item())

	pear_bosc_indices = find_indices(fruit_labels, "Pear Bosc")
	pear_williams_indices = find_indices(fruit_labels, "Pear Williams")
	avo_empire_indices = find_indices(fruit_labels, "Avocado Emp")
	avo_organic_indices = find_indices(fruit_labels, "Avocado Organic")

	y_true_labels = np.asarray(y_true_labels)
	y_pred_labels = np.asarray(y_pred_labels)
	y_true_sublabels = np.asarray(y_true_sublabels)
	y_pred_sublabels = np.asarray(y_pred_sublabels)

	accuracy_labels = 100. * (running_correct_labels / len(test_data_loader.dataset))
	accuracy_sublabels = 100. * (running_correct_sublabels / len(test_data_loader.dataset))

	classification_evaluate(y_true_labels, y_pred_labels, "all")
	classification_evaluate(y_true_labels[pear_bosc_indices], y_pred_labels[pear_bosc_indices], "pear_bosc")
	classification_evaluate(y_true_labels[pear_williams_indices], y_pred_labels[pear_williams_indices], "pear_williams")
	classification_evaluate(y_true_labels[avo_empire_indices], y_pred_labels[avo_empire_indices], "avocado_emp")
	classification_evaluate(y_true_labels[avo_organic_indices], y_pred_labels[avo_organic_indices], "avocado_organic")

	classification_evaluate(y_true_sublabels, y_pred_sublabels, "all_sublabels", labels_dict=TIME_LEFT_DICT)
	classification_evaluate(y_true_sublabels[pear_bosc_indices], y_pred_sublabels[pear_bosc_indices], "pear_bosc_sublabels", labels_dict=TIME_LEFT_DICT)
	classification_evaluate(y_true_sublabels[pear_williams_indices], y_pred_sublabels[pear_williams_indices], "pear_williams_sublabels", labels_dict=TIME_LEFT_DICT)
	classification_evaluate(y_true_sublabels[avo_empire_indices], y_pred_sublabels[avo_empire_indices], "avocado_emp_sublabels", labels_dict=TIME_LEFT_DICT)
	classification_evaluate(y_true_sublabels[avo_organic_indices], y_pred_sublabels[avo_organic_indices], "avocado_organic_sublabels", labels_dict=TIME_LEFT_DICT)

	return (losses.avg, losses_class.avg, losses_subclass.avg), (accuracy_labels, accuracy_sublabels)

def find_indices(list, fruit):
	return [i for i, x in enumerate(list) if x == fruit]

def get_ovr_roc():
	""" Gets the ROC curve for One vs Rest classification """
	pass

import textwrap
def wrap_labels(ax, width, break_long_words=False):
	labels = []
	for label in ax.get_xticklabels():
		text = label.get_text()
		labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
	ax.set_xticklabels(labels, rotation=0)
	ax.set_yticklabels(labels, rotation=90, horizontalalignment="center")
	ax.tick_params(axis="y", which="major", pad=15)

def classification_evaluate(y_true, y_pred, title, labels_dict=LABELS_DICT):
	confusion_mat = confusion_matrix(y_true, y_pred)
	df_confusion_mat = pd.DataFrame(confusion_mat / np.sum(confusion_mat, axis=1)[:, None], index = [key for key, value in labels_dict.items()], columns = [key for key, value in labels_dict.items()])
	fig, ax = plt.subplots(figsize=(10, 10))
	sns.heatmap(df_confusion_mat, annot=True, fmt=".2%", cmap="Blues")
	wrap_labels(ax, 10) if title.split("_")[-1] == "sublabels" else None
	print("Title: {}, Accuracy: {}".format(title, accuracy_score(y_true, y_pred)))
	print(classification_report(y_true, y_pred, target_names=[key for key, value in labels_dict.items()]))
	plt.tight_layout()
	# print(df_confusion_mat)
	plt.savefig(os.path.join(VISUALIZATION_DIR_NAME, "confusion_matrix_{}.pdf".format(title)))
	plt.show()
	plt.close()

if __name__ == "__main__":
	create_directory(os.path.join(VISUALIZATION_DIR_NAME))
	create_directory(os.path.join(MODEL_PATH, "classification"))
	# main()
	test_model_only()