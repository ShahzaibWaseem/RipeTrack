import os, sys
sys.path.append(os.path.join(".."))

import json
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from config import TEST_DATASETS

def main():	
	X, y = np.empty(shape=(1, 256), dtype=np.int32), np.empty(shape=(1,), dtype=np.int8)

	with open(os.path.join("..", "inference", "weights_RGBN.json"), "r") as jsonFile:
		data = json.load(jsonFile)
		for idx, dict in enumerate(data):
			weights = np.array(dict["%d"%idx]["output"])
			labels = dict["%d"%idx]["label"]
			X = np.append(X, weights, axis=0)
			y = np.append(y, labels)

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
	plt.savefig(os.path.join("..", "inference", "cluster_RGBN.png"))
	plt.show()

if __name__ == "__main__":
	main()