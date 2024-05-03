""" Deprecated: file will be deleted in the future """
import os
import time
import pickle

import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_validate

from dataset import get_dataloaders_classification
from config import VISUALIZATION_DIR_NAME, MODEL_PATH, LABELS_DICT, TIME_LEFT_DICT

import seaborn as sns
import matplotlib.pyplot as plt

def fit_model(model, X_train, y_train, X_test, y_test, model_name, labels_dict=LABELS_DICT):
	start = time.time()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	print("\n", model_name, ":")
	print("Accuracy:", accuracy_score(y_test, y_pred))
	print("Time taken:", time.time() - start)
	confusion_mat = confusion_matrix(y_test, y_pred)
	df_confusion_mat = pd.DataFrame(confusion_mat/np.sum(confusion_mat, axis=1)[:, None], index=[key for key, _ in labels_dict.items()], columns=[key for key, _ in labels_dict.items()])
	plt.figure(figsize=(10, 10))
	sns.heatmap(df_confusion_mat, annot=True, fmt=".2%")
	print(classification_report(y_test, y_pred, target_names=[key for key, _ in labels_dict.items()]))
	print(df_confusion_mat)
	plt.savefig(os.path.join(VISUALIZATION_DIR_NAME, "confusion_matrix_{}.png".format(model_name)))
	plt.show()
	plt.close()

def plot_losses(model, filename):
	plt.plot(model.loss_curve_)
	plt.xlabel("Iterations")
	plt.ylabel("Loss")
	plt.title("Loss Curve")
	plt.show()
	plt.savefig(os.path.join(VISUALIZATION_DIR_NAME, filename + "_loss.png"))

def get_data(train_data_loader, valid_data_loader):
	X, y = [], []
	stride = 10
	for hypercubes, labels, _ in train_data_loader:
		hypercubes = hypercubes.squeeze().numpy()
		bands, height, width = hypercubes.shape
		# print(bands, height, width, hypercubes.shape)
		hypercubes_sig = hypercubes[:, 0:height:stride, 0:width:stride]
		for i in range(0, hypercubes_sig.shape[1]):
			for j in range(0, hypercubes_sig.shape[2]):
				X.append(hypercubes_sig[:, i, j])
				y.append(labels.ravel().numpy())

	for hypercubes, labels, _ in valid_data_loader:
		hypercubes = hypercubes.squeeze().numpy()
		bands, height, width = hypercubes.shape
		hypercubes_sig = hypercubes[:, 0:height:stride, 0:width:stride]
		for i in range(0, hypercubes_sig.shape[1]):
			for j in range(0, hypercubes_sig.shape[2]):
				X.append(hypercubes_sig[:, i, j])
				y.append(labels.ravel().numpy())

	return np.asarray(X), np.asarray(y).ravel()

def get_data_batch(train_data_loader, valid_data_loader):
	X, y, y_sublabels, fruits = [], [], [], []
	stride = 10
	for hypercubes, labels, sublabels, fruit_labels in train_data_loader:
		hypercubes = hypercubes.squeeze().numpy()
		batch, bands, height, width = hypercubes.shape
		for batch_idx in range(0, batch):
			X.append(hypercubes[batch_idx].flatten())
			y.append(labels[batch_idx].ravel().numpy())
			y_sublabels.append(sublabels[batch_idx].ravel().numpy())
		fruits.extend(fruit_labels)

	for hypercubes, labels, sublabels, fruit_labels in valid_data_loader:
		hypercubes = hypercubes.squeeze().numpy()
		for batch_idx in range(0, batch):
			X.append(hypercubes[batch_idx].flatten())
			y.append(labels[batch_idx].ravel().numpy())
			y_sublabels.append(sublabels[batch_idx].ravel().numpy())
		fruits.extend(fruit_labels)

	return np.asarray(X), np.asarray(y).ravel(), np.asarray(y_sublabels).ravel(), fruits

def main():
	train_data_loader, valid_data_loader = get_dataloaders_classification()
	X, y = get_data(train_data_loader, valid_data_loader)

	print("X Shape: {}, y Shape: {}".format(X.shape, y.shape))

	skf = StratifiedKFold(n_splits=4, random_state=42, shuffle=True)
	scaler = MinMaxScaler()
	# mlp = MLPClassifier(hidden_layer_sizes=(200, 150, 100), max_iter=300, activation="relu", solver="adam", alpha=0.0001, verbose=True, n_jobs=1)
	logR = LogisticRegression(n_jobs=4, class_weight="balanced", tol=1e-7, penalty="elasticnet", multi_class="multinomial", solver="saga", verbose=True, l1_ratio=0.5)
	pipeline = Pipeline([("scaler", scaler), ("logr", logR)])
	for k, (train_index, val_index) in enumerate(skf.split(X, y)):
		X_train, X_test = X[train_index], X[val_index]
		y_train, y_test = y[train_index], y[val_index]
		print("Fold: {}, X Train: {}, y Train: {}, X Test: {}, y Test: {}".format(k, X_train.shape, y_train.shape, X_test.shape, y_test.shape))
		fit_model(pipeline, X_train, y_train, X_test, y_test, "logR" + str(k), k)
		# plot_losses(logR, "LogR_" + str(k))
		pickle.dump(pipeline, open(os.path.join(MODEL_PATH, "LogR_" + str(k) + ".pkl"), "wb"))

def main2():
	logR = LogisticRegression(n_jobs=4, class_weight="balanced", tol=1e-7, penalty="elasticnet", multi_class="multinomial", solver="saga", verbose=True, l1_ratio=0.5)
	svm = SVC(kernel="rbf", class_weight="balanced", probability=True, verbose=True)
	sgd = SGDClassifier(loss="log", penalty="elasticnet", alpha=0.0001, l1_ratio=0.5, max_iter=500, tol=1e-7, verbose=True, n_jobs=4, class_weight="balanced")

	test_data_loader, valid_data_loader = get_dataloaders_classification(trainset_size=1.0)
	X, y, y_sublabels, fruits = get_data_batch(test_data_loader, valid_data_loader)
	print("X Shape: {}, y Labels: {}, y SubLabels: {} Fruits: {}".format(X.shape, y.shape, y_sublabels.shape, len(fruits)))

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)
	print("Labels\t\tX Train: {}, X Test: {}, y Train: {}, y Test: {}".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

	X_sublabels_train, X_sublabels_test, y_sublabels_train, y_sublabels_test = train_test_split(X, y_sublabels, test_size=0.15, random_state=42, shuffle=True)
	print("SubLabels\tX Train: {}, X Test: {}, y Train: {}, y Test: {}".format(X_sublabels_train.shape, X_sublabels_test.shape, y_sublabels_train.shape, y_sublabels_test.shape))

	# fit_model(mlp, X_sublabels_train, y_sublabels_train, X_sublabels_test, y_sublabels_test, "MLP", labels_dict=TIME_LEFT_DICT)
	fit_model(logR, X_sublabels_train, y_sublabels_train, X_sublabels_test, y_sublabels_test, "LogisticRegression", labels_dict=TIME_LEFT_DICT)
	fit_model(svm, X_sublabels_train, y_sublabels_train, X_sublabels_test, y_sublabels_test, "SVM", labels_dict=TIME_LEFT_DICT)
	fit_model(sgd, X_sublabels_train, y_sublabels_train, X_sublabels_test, y_sublabels_test, "SGD", labels_dict=TIME_LEFT_DICT)

	# fit_model(mlp, X_train, y_train, X_test, y_test, "MLP")
	# fit_model(logR, X_train, y_train, X_test, y_test, "LogisticRegression")
	fit_model(sgd, X_train, y_train, X_test, y_test, "SGD")
	fit_model(svm, X_train, y_train, X_test, y_test, "SVM")

if __name__ == "__main__":
	main2()