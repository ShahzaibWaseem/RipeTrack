import os
import time
import pickle
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from dataset import DatasetFromDirectory, get_dataloaders_classification
from config import VISUALIZATION_DIR_NAME, MODEL_PATH, LABELS_DICT

def fit_model(model, X_train, y_train, X_test, y_test, model_name, k):
	start = time.time()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	print("\n", model_name, ":")
	print("Accuracy:", accuracy_score(y_test, y_pred))
	print("Time taken:", time.time() - start)
	confusion_mat = confusion_matrix(y_test, y_pred)
	df_confusion_mat = pd.DataFrame(confusion_mat / np.sum(confusion_mat, axis=1)[:, None], index = [key for key, value in LABELS_DICT.items()], columns = [key for key, value in LABELS_DICT.items()])
	plt.figure()
	sns.heatmap(df_confusion_mat, annot=True, fmt=".2%")
	print(classification_report(y_test, y_pred, target_names=[key for key, value in LABELS_DICT.items()]))
	print(df_confusion_mat)
	plt.savefig(os.path.join(VISUALIZATION_DIR_NAME, "confusion_matrix_{}.png".format(k)))
	plt.show()

def plot_losses(model, filename):
	plt.plot(model.loss_curve_)
	plt.xlabel("Iterations")
	plt.ylabel("Loss")
	plt.title("Loss Curve")
	plt.show()
	plt.savefig(os.path.join(VISUALIZATION_DIR_NAME, filename + "_loss.png"))

def get_data(train_data_loader, valid_data_loader):
	X, y = [], []
	stride = 5
	for hypercubes, labels, _ in train_data_loader:
		hypercubes = hypercubes.squeeze().numpy()
		bands, height, width = hypercubes.shape
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

if __name__ == "__main__":
	main()