import os
import time
import argparse
from tqdm import tqdm
from collections import Counter

import numpy as np
import pandas as pd

import torch
from torchsummary import summary
from torch.autograd import Variable
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, accuracy_score, confusion_matrix, classification_report,\
	auc, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

from models.classifier import ModelWithAttention

from dataset import get_dataloaders_classification
from utils import AverageMeter, create_directory, initialize_logger, save_checkpoint, get_best_checkpoint
from config import VISUALIZATION_DIR_NAME, MODEL_PATH, TEST_DATASETS, BANDS, LABELS_DICT, TIME_LEFT_DICT,\
	end_epoch, classicication_run_title, run_pretrained, confusion_font_dict

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.rc("font", **confusion_font_dict)

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
	# checkpoint_filename, epoch, iter, state_dict, optimizer, val_loss, (val_acc_labels, val_acc_sublabels) = get_best_checkpoint(task="classification")
	checkpoint_filename = "RT_ModelWithAttention_shelflife_030 transferLearning on new Fruits [Reconstructed].pkl"
	checkpoint = torch.load(os.path.join(MODEL_PATH, "classification", "others", checkpoint_filename))
	epoch, iter, state_dict, opt_state, val_loss, (val_acc_labels, val_acc_sublabels) = checkpoint["epoch"], checkpoint["iter"], checkpoint["state_dict"],\
		checkpoint["optimizer"], checkpoint["val_loss"], checkpoint["val_acc"]
	model = ModelWithAttention(input_channels=len(BANDS), num_classes=len(LABELS_DICT), num_subclasses=len(TIME_LEFT_DICT))
	model = model.cuda()
	model.eval()
	summary(model=model, input_data=(68, 512, 512))
	criterion = (torch.nn.CrossEntropyLoss(reduction="mean"), torch.nn.CrossEntropyLoss(reduction="mean"))
	model.load_state_dict(state_dict)
	test_data_loader, _ = get_dataloaders_classification(trainset_size=1.0)
	test_loss, test_acc = test(test_data_loader, model, criterion)
	print("Test Loss: {}, Test Accuracy: {}".format(test_loss, test_acc))

def main():
	logger = initialize_logger(filename="classification.log")
	history = {"train_loss": [], "train_loss_labels": [], "train_loss_sublabels": [], "train_acc_labels": [], "train_acc_sublabels": [], "val_loss": [], "val_loss_labels": [], "val_loss_sublabels": [], "val_acc_labels": [], "val_acc_sublabels": []}
	log_string = "Epoch [%3d], Iter[%7d], Time: %.2f, Train Loss: %.8f (%.8f, %.8f), Train Accuracy: %.2f%%, %.2f%%, Validation Loss: %.8f (%.8f, %.8f), Validation Accuracy: %.2f%%, %.2f%%"

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

	model = ModelWithAttention(input_channels=len(BANDS), num_classes=len(LABELS_DICT), num_subclasses=len(TIME_LEFT_DICT))
	# model.bottleneck.register_forward_hook(get_activation("bottleneck"))
	model = model.cuda()
	# summary(model=model, input_data=(68, 512, 512))

	criterion_class = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
	criterion_subclass = torch.nn.CrossEntropyLoss()
	criterion = (criterion_class, criterion_subclass)
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, amsgrad=True, betas=(0.9, 0.999), weight_decay=1e-6)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5, verbose=True)

	epoch, iteration, best_epoch, best_val_loss, best_val_acc_labels, best_val_acc_sublabels = 0, 0, 0, 0, 0, 0

	if run_pretrained:
		# checkpoint_filename, epoch, iteration, state_dict, optimizer, val_loss, (val_acc_labels, val_acc_sublabels) = get_best_checkpoint(task="classification")
		checkpoint_filename = "RT_ModelWithAttention_shelflife_100 final trained on 4 fruits.pkl"
		checkpoint = torch.load(os.path.join(MODEL_PATH, "classification", "others", checkpoint_filename))
		epoch, iter, state_dict, opt_state, val_loss, (val_acc_labels, val_acc_sublabels) = checkpoint["epoch"], checkpoint["iter"], checkpoint["state_dict"],\
			checkpoint["optimizer"], checkpoint["val_loss"], checkpoint["val_acc"]
		model.load_state_dict(state_dict)
		optimizer.load_state_dict(opt_state)
		start_epoch = epoch
		print("Loaded model from checkpoint: Filename: %s Epochs Run: %d, Validation Loss: %.9f" % (checkpoint_filename, epoch, val_loss))

	start_epoch = 1

	for epoch in range(start_epoch, end_epoch):
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
		if epoch % 10 == 0:
				save_checkpoint(int(round(epoch, -1)), iteration_passed, best_model, best_optimizer, best_val_loss, best_val_acc_labels, best_val_acc_sublabels, bands=BANDS, task="classification")
		if epoch % 50 == 0:
			test_loss, test_acc = test(valid_data_loader, best_model, criterion)
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
	plt.plot(history["train_acc_sublabels"], label="Train SubLabel")
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
	y_pred_labels, y_true_labels, y_pred_sublabels, y_true_sublabels, y_pred_labels_proba, y_pred_sublabels_proba, fruit_labels = [], [], [], [], [], [], []

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

		preds_labels_proba = torch.nn.functional.softmax(out_labels, dim=1)
		preds_sublabels_proba = torch.nn.functional.softmax(out_sublabels, dim=1)

		_, preds_labels = torch.max(out_labels.data, 1)
		_, preds_sublabels = torch.max(out_sublabels.data, 1)

		loss_class = criterion_class(out_labels, labels)
		loss_subclass = criterion_subclass(out_sublabels, sublabels)
		loss = loss_class + loss_subclass

		running_correct_labels += (preds_labels == labels).sum().item()
		running_correct_sublabels += (preds_sublabels == sublabels).sum().item()

		y_pred_labels.extend(preds_labels.data.cpu().numpy())
		y_pred_sublabels.extend(preds_sublabels.data.cpu().numpy())
		y_pred_labels_proba.extend(preds_labels_proba.data.cpu().numpy())
		y_pred_sublabels_proba.extend(preds_sublabels_proba.data.cpu().numpy())
		fruit_labels.extend(fruits)

		losses.update(loss.item())
		losses_class.update(loss_class.item())
		losses_subclass.update(loss_subclass.item())

	y_true_labels = np.asarray(y_true_labels)
	y_pred_labels = np.asarray(y_pred_labels)
	y_true_sublabels = np.asarray(y_true_sublabels)
	y_pred_sublabels = np.asarray(y_pred_sublabels)
	y_pred_labels_proba = np.asarray(y_pred_labels_proba)
	y_pred_sublabels_proba = np.asarray(y_pred_sublabels_proba)

	accuracy_labels = 100. * (running_correct_labels / len(test_data_loader.dataset))
	accuracy_sublabels = 100. * (running_correct_sublabels / len(test_data_loader.dataset))

	classification_evaluate(y_true_labels, y_pred_labels, "all", acc=accuracy_labels)
	classification_evaluate(y_true_sublabels, y_pred_sublabels, "all_sublabels", labels_dict=TIME_LEFT_DICT, acc=accuracy_sublabels)

	for fruit in TEST_DATASETS:
		fruit_fullname = " ".join(elem.capitalize() for elem in fruit.split("-"))
		fruit_indices = find_indices(fruit_labels, fruit_fullname)
		print(fruit_fullname, fruit)
		classification_evaluate(y_true_labels[fruit_indices], y_pred_labels[fruit_indices], fruit)
		classification_evaluate(y_true_sublabels[fruit_indices], y_pred_sublabels[fruit_indices], fruit + "_sublabels", labels_dict=TIME_LEFT_DICT)

	label_binarizer = LabelBinarizer().fit(y_true_labels)
	y_onehot_labels_test = label_binarizer.transform(y_true_labels)
	print("Labels:\t\t{}\t{}".format(y_onehot_labels_test.shape, y_pred_labels_proba.shape))

	label_binarizer = LabelBinarizer().fit(y_true_sublabels)
	y_onehot_sublabels_test = label_binarizer.transform(y_true_sublabels)
	print("SubLabels:\t{}\t{}".format(y_onehot_sublabels_test.shape, y_pred_sublabels_proba.shape))

	get_ovr_roc(y_onehot_labels_test, y_pred_labels_proba)
	get_ovr_roc(y_onehot_sublabels_test, y_pred_sublabels_proba, labels_dict=TIME_LEFT_DICT)
	pr_auc_curve(y_onehot_labels_test, y_pred_labels_proba, y_true_labels)
	pr_auc_curve(y_onehot_sublabels_test, y_pred_sublabels_proba, y_true_sublabels, labels_dict=TIME_LEFT_DICT)

	return (losses.avg, losses_class.avg, losses_subclass.avg), (accuracy_labels, accuracy_sublabels)

def find_indices(list, fruit):
	return [i for i, x in enumerate(list) if x == fruit]

def pr_auc_curve(y_true_labels, y_pred_labels_proba, y_test, labels_dict=LABELS_DICT):
	""" Gets the Precision Recall curve for One vs Rest classification """
	nclasses = len(labels_dict)
	precision, recall, thresholds, fScores, average_precision = dict(), dict(), dict(), dict(), dict()
	for i in range(nclasses):
		precision[i], recall[i], thresholds[i] = precision_recall_curve(y_true_labels[:, i], y_pred_labels_proba[:, i])
		fScores[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])
		average_precision[i] = average_precision_score(y_true_labels[:, i], y_pred_labels_proba[:, i])
	# A "micro-average": quantifying score on all classes jointly
	precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_labels.ravel(), y_pred_labels_proba.ravel())
	average_precision["micro"] = average_precision_score(y_true_labels, y_pred_labels_proba, average="micro")

	# setup plot details
	colors = ["r", "b", "g", "k", "r", "b", "g", "k", "r", "b", "g"]
	linestyles = ["solid", "solid", "solid", "solid", "dashdot", "dashdot", "dashdot", "dashed", "dashed", "dashed", "dotted"]
	_, ax = plt.subplots(figsize=(13, 10))

	f_scores = np.linspace(0.2, 0.8, num=4)
	lines, labels = [], []
	for f_score in f_scores:
		x = np.linspace(0.01, 1)
		y = f_score * x / (2 * x - f_score)
		# (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
		# plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

	# display = PrecisionRecallDisplay(recall=recall["micro"], precision=precision["micro"], average_precision=average_precision["micro"], prevalence_pos_label=Counter(y_test)[1] / len(y_test))
	# display.plot(ax=ax, name="μ-avg P-R", color="gold", linestyle=":", plot_chance_level=True, linewidth=2.5)
	linestyles_idx, color_idx = 0, 0
	for i in range(nclasses):
		print("Class: {}\tPrecision: {}\tRecall: {}\tF1: {}\tThresholds: {}".format(list(labels_dict.keys())[i], precision[i], recall[i], fScores[i], thresholds[i]))
		max_idx = np.argmax(fScores[i])
		print("Best Threshold: {}\t FScore: {}\tPrecision: {}\tRecall: {}".format(thresholds[i][max_idx], fScores[i][max_idx], precision[i][max_idx], recall[i][max_idx]))
		display = PrecisionRecallDisplay(recall=recall[i], precision=precision[i])
		display.plot(ax=ax, name=f"{list(labels_dict.keys())[i]}", color=colors[color_idx], linestyle=linestyles[linestyles_idx], linewidth=2.5)
		ax.scatter(recall[i][max_idx], precision[i][max_idx], marker="o", color=colors[color_idx], linewidth=4)
		linestyles_idx = (linestyles_idx + 1) % len(linestyles)
		color_idx = (color_idx + 1) % len(colors)

	# add the legend for the iso-f1 curves
	# handles, labels = display.ax_.get_legend_handles_labels()
	# handles.extend([l])
	# labels.extend(["iso-f1 curves"])
	# set the legend and the axes
	ax.set_xlim([0.0, 1.0])
	ax.set_ylim([0.0, 1.05])
	# ax.legend(handles=handles, labels=labels, loc="best")
	# ax.set_title("Precision-Recall curve for {}Classes".format("Sub-" if labels_dict == TIME_LEFT_DICT else ""))
	plt.tight_layout(pad=0)
	plt.savefig(os.path.join(VISUALIZATION_DIR_NAME, "pr_auc_curve_{}.pdf".format("subclasses" if labels_dict == TIME_LEFT_DICT else "classes")))
	plt.show()
	plt.close()

def get_ovr_roc(y_true_labels, y_pred_labels_proba, labels_dict=LABELS_DICT):
	""" Gets the ROC curve for One vs Rest classification """
	nclasses = len(labels_dict)
	fpr, tpr, roc_auc = dict(), dict(), dict()
	fpr["micro"], tpr["micro"], _ = roc_curve(y_true_labels.ravel(), y_pred_labels_proba.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	for i in range(nclasses):
		fpr[i], tpr[i], _ = roc_curve(y_true_labels[:, i], y_pred_labels_proba[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	fpr_grid = np.linspace(0.0, 1.0, 1000)

	# Interpolate all ROC curves at these points
	mean_tpr = np.zeros_like(fpr_grid)

	for i in range(nclasses):
		mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

	# Average it and compute AUC
	mean_tpr /= nclasses

	fpr["macro"] = fpr_grid
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")

	macro_roc_auc_ovr = roc_auc_score(y_true_labels, y_pred_labels_proba, multi_class="ovr", average="macro")

	print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{macro_roc_auc_ovr:.2f}")

	fig, ax = plt.subplots(figsize=(13, 10))

	# plt.plot(fpr["micro"], tpr["micro"], label=f"μ-avg ROC (AUC = {roc_auc['micro']:.2f})", color="deeppink", linestyle=":", linewidth=4)
	# plt.plot(fpr["macro"], tpr["macro"], label=f"μ-avg ROC (AUC = {roc_auc['macro']:.2f})", color="navy", linestyle=":", linewidth=4)

	linestyles_idx, color_idx = 0, 0
	colors = ["r", "b", "g", "k", "r", "b", "g", "k", "r", "b", "g"]
	linestyles = ["solid", "solid", "solid", "solid", "dashdot", "dashdot", "dashdot", "dashed", "dashed", "dashed", "dotted"]

	for class_id in range(nclasses):
		RocCurveDisplay.from_predictions(y_true_labels[:, class_id], y_pred_labels_proba[:, class_id],
			name=f"{list(labels_dict.keys())[class_id]}", color=colors[color_idx], linestyle=linestyles[linestyles_idx], ax=ax, plot_chance_level=False, linewidth=2.5)
		linestyles_idx = (linestyles_idx + 1) % len(linestyles)
		color_idx = (color_idx + 1) % len(colors)

	plt.axis("square")
	plt.xlabel("False Positive Rate", **confusion_font_dict)
	plt.ylabel("True Positive Rate", **confusion_font_dict)
	# plt.title("Receiver Operating Characteristic curve for {}Classes\n(One-vs-Rest)".format("Sub-" if labels_dict == TIME_LEFT_DICT else ""))
	plt.legend()
	plt.tight_layout(pad=0)
	plt.savefig(os.path.join(VISUALIZATION_DIR_NAME, "roc_ovr_curve_{}.pdf".format("subclasses" if labels_dict == TIME_LEFT_DICT else "classes")))
	plt.show()
	plt.close()

import textwrap
def wrap_labels(ax, width, break_long_words=False):
	labels = []
	for label in ax.get_xticklabels():
		text = label.get_text()
		labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
	ax.set_xticklabels(labels, rotation=0)
	ax.set_yticklabels(labels, rotation=90, horizontalalignment="center")
	ax.tick_params(axis="y", which="major", pad=15)

def classification_evaluate(y_true, y_pred, title, labels_dict=LABELS_DICT, acc=0.0):
	# confusion_mat = confusion_matrix(y_true, y_pred)
	# df_confusion_mat = pd.DataFrame(confusion_mat / np.sum(confusion_mat, axis=1)[:, None], index = [key for key, value in labels_dict.items()], columns = [key for key, value in labels_dict.items()])
	# fig, ax = plt.subplots(figsize=(13, 10))
	# sns.heatmap(round(df_confusion_mat * 100, 0), annot=True, cmap="Blues" if labels_dict == LABELS_DICT else "Oranges")
	# wrap_labels(ax, 10) if title.split("_")[-1] == "sublabels" else None
	print("Title: {}, Accuracy: {}, {}".format(title, accuracy_score(y_true, y_pred), acc))
	# print(classification_report(y_true, y_pred, target_names=[key for key, value in labels_dict.items()]))
	# plt.tight_layout()
	# print(df_confusion_mat)
	# plt.savefig(os.path.join(VISUALIZATION_DIR_NAME, "confusion_matrix_{}.pdf".format(title)))
	# plt.show()
	# plt.close()


if __name__ == "__main__":
	create_directory(os.path.join(VISUALIZATION_DIR_NAME))
	create_directory(os.path.join(MODEL_PATH, "classification"))
	# main()
	test_model_only()