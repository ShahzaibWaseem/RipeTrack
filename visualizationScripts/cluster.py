""" Depreciated: file will be removed in the future """
import os
import sys
sys.path.append(os.path.join(".."))

import json
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch

from torch_classifier import test
from models.classifier import ModelWithAttention

from dataset import get_dataloaders
from utils import get_best_checkpoint, get_activation
from config import TEST_DATASETS, EXTRACT_DATASETS, BANDS, predef_input_transform, predef_label_transform

import matplotlib.pyplot as plt

def get_json(test_model=True):
	X, y, actual_labels = np.empty(shape=(1, 256), dtype=np.int32), np.empty(shape=(1,), dtype=np.int8), np.empty(shape=(1,), dtype=np.int8)

	if test_model:
		model = ModelWithAttention(fine_tune=True, in_channels=len(BANDS), num_classes=len(TEST_DATASETS))
		model.bottleneck.register_forward_hook(get_activation("bottleneck"))
		model = model.cuda()

		criterion = torch.nn.CrossEntropyLoss()
		_, _, test_data_loader = get_dataloaders(predef_input_transform, predef_label_transform, task="classification")
		epoch, iter, state_dict, optimizer, val_loss, val_acc = get_best_checkpoint(task="classification")
		model.load_state_dict(state_dict)

		test_loss, test_acc, json_data = test(test_data_loader, model, criterion)
		print("Test Loss: %.9f, Test Accuracy: %.2f%%" % (test_loss, test_acc))
	else:
		with open(os.path.join("inference", "weights_HS.json"), "r") as json_file:
			json_data = json.load(json_file)

	for dict in json_data:
		weights = np.array(dict["output"])
		labels = dict["label"]
		actual_label = dict["actual_label"]
		X = np.append(X, weights, axis=0)
		y = np.append(y, labels)
		actual_labels = np.append(actual_labels, actual_label)
	return X, y, actual_labels

def main():
	X, y, actual_labels = get_json(test_model=False)
	X = StandardScaler().fit_transform(X)					# scaling the data

	pca = PCA(3)											# reducing the dimensions to 3
	X = pca.fit_transform(X)
	
	X = np.delete(X, 0, axis=0)								# removing an outlier
	y = np.delete(y, 0, axis=0)
	actual_labels = np.delete(actual_labels, 0, axis=0)

	# Clustering. Just to get centroid, clustering not used
	kmeans = KMeans(init="random", n_clusters=2, random_state=32, max_iter=1000).fit(X)
	centroids = kmeans.cluster_centers_

	svm = SVC(kernel="linear", C=1.0).fit(X, y)				# SVM to get the decision boundary (for plot)

	# Getting the decision boundary plane
	x_mesh, y_mesh = np.meshgrid(np.linspace(-5, 5, 30), np.linspace(-5, 5, 30))
	z = lambda x, y: (-svm.intercept_[0]-svm.coef_[0][0]*x -svm.coef_[0][1]*y) / svm.coef_[0][2]
	z_plane = np.clip(z(x_mesh, y_mesh), -2.25, 2.25)
	
	# Plotting
	fig = plt.figure(figsize=(20, 10))
	ax = plt.axes(projection="3d")

	for i in range(len(EXTRACT_DATASETS)):
		ax.scatter3D(X[actual_labels==i, 0], X[actual_labels==i, 1], X[actual_labels==i, 2], label=EXTRACT_DATASETS[i])	# scattered elements
	ax.scatter3D(centroids[:,0], centroids[:,1], centroids[:,2], s=80, color = "black")									# centroids
	ax.plot_surface(x_mesh, y_mesh, z_plane, alpha=0.2)		# decision boundary plane

	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	ax.view_init(10, 10)
	plt.legend()
	plt.savefig(os.path.join("inference", "cluster_HS.png"))
	plt.show()

if __name__ == "__main__":
	os.chdir("..")											# go up a directory as this script is in the visualization folder
	main()