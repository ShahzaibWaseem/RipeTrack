import os, sys
sys.path.append(os.path.join(".."))

import json
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch

import matplotlib.pyplot as plt

from utils import get_best_checkpoint
from torch_classifier import TorchClassifier, get_loaders, test, get_activation
from config import BANDS, TEST_DATASETS, predef_input_transform, predef_label_transform, init_lr, run_pretrained

def get_json(test_model=True):
	X, y = np.empty(shape=(1, 256), dtype=np.int32), np.empty(shape=(1,), dtype=np.int8)

	if test_model:
		model = TorchClassifier(fine_tune=True, in_channels=len(BANDS), num_classes=len(TEST_DATASETS))
		model.bottleneck.register_forward_hook(get_activation("bottleneck"))
		model = model.cuda()

		criterion = torch.nn.CrossEntropyLoss()
		_, _, test_data_loader = get_loaders(predef_input_transform, predef_label_transform)
		epoch, iter, state_dict, optimizer, val_loss, val_acc = get_best_checkpoint(task="classification")
		model.load_state_dict(state_dict)

		test_loss, test_acc, json_data = test(test_data_loader, model, criterion)
		print("Test Loss: %.9f, Test Accuracy: %.2f%%" % (test_loss, test_acc))
	else:
		with open(os.path.join("..", "inference", "weights_HS.json"), "r") as json_file:
			json_data = json.load(json_file)

	for idx, dict in enumerate(json_data):
		weights = np.array(dict["%d"%idx]["output"])
		labels = dict["%d"%idx]["label"]
		X = np.append(X, weights, axis=0)
		y = np.append(y, labels)
	return X, y

def main():
	get_best_checkpoint(task="classification", up_a_directory=True)
	X, y = get_json(test_model=True)

	# scaling the data
	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	# reducing the dimension
	pca = PCA(3)
	pca_X = pca.fit_transform(X)
	# removing an outlier
	pca_X = np.delete(pca_X, 0, axis=0)
	y = np.delete(y, 0, axis=0)

	# clustering
	kmeans = KMeans(init="random", n_clusters=2, random_state=32, max_iter=1000)
	label = kmeans.fit_predict(pca_X)
	centroids = kmeans.cluster_centers_
	u_labels = np.unique(y)

	fig = plt.figure(figsize=(20, 10))
	ax = plt.axes(projection="3d")

	for i in u_labels:
		ax.scatter3D(pca_X[y==i, 0], pca_X[y==i, 1], pca_X[y==i, 2], label=TEST_DATASETS[i])
	ax.scatter3D(centroids[:,0], centroids[:,1], centroids[:,2], s=80, color = "black")
	# ax.view_init(-140, 60)
	plt.legend()
	plt.savefig(os.path.join("..", "inference", "cluster_HS.png"))
	plt.show()

if __name__ == "__main__":
	main()