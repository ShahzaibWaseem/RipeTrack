import itertools
import os
import numpy as np

import h5py
from glob import glob
from imageio import imread
from utils import load_mat

import torch
from torch.utils.data import Dataset
from config import var_name, BAND_SPACING, RGBN_BANDS

class DatasetFromHdf5(Dataset):
	def __init__(self, file_path):
		super(DatasetFromHdf5, self).__init__()
		hf = h5py.File(file_path)
		self.images = hf.get("data")
		self.ground_t = hf.get("label")

	def __getitem__(self, index):
		return torch.from_numpy(self.images[index,:,:,:]).float(), torch.from_numpy(self.ground_t[index,:,:,:]).float()

	def __len__(self):
		return len(self.images)

def image_to_patches(image, patch_size, discard_edges=True):
	""" 
	Splits the image into patches and returns a list of patches
		image: (RGB-NIR or Hypercube) numpy array of shape (channel, row, col)
		discard_edges: if True, discard the four corners of the image
	"""
	patches = []
	for i in range(0, image.shape[1] - patch_size + 1, patch_size):
		for j in range(0, image.shape[2] - patch_size + 1, patch_size):
			if discard_edges and (i == 0 or i == image.shape[0] - patch_size or j == 0 or j == image.shape[1] - patch_size):
				continue
			patches.append(image[:, i:i+patch_size, j:j+patch_size])
	return patches

def read_image(rgb_filename, nir_filename):
	""" Reads the two images and stack them together while maintaining the order or BGR-NIR """
	rgb = imread(rgb_filename)
	rgb = rgb/255
	rgb[:,:, [0, 2]] = rgb[:,:, [2, 0]]		# flipping red and blue channels (shape used for training)

	nir = imread(nir_filename)[:,:, 0]
	nir = nir/255

	image = np.dstack((rgb, nir))
	image = np.transpose(image, [2, 0, 1])
	del rgb, nir

	return image

class DatasetDirectoryProductPairing(Dataset):
	# Expects the directory structure to be:
	# root/
	# 	category1/		(oats, flour, etc.)
	# 		label1/		(gluten)
	# 			01_RGB.jpg
	# 			01_NIR.jpg
	# 			...
	# 		label2/		(glutenfree)
	# 			01_RGB.jpg
	# 			01_NIR.jpg
	# 			...
	# 	...
	IMAGE_SIZE = 512
	images, labels = {}, {}

	def __init__(self, root, dataset_name=None, train_with_patches=True, permute_data=True, patch_size=64, discard_edges=True):
		self.PATCH_SIZE = patch_size
		self.root = root
		self.var_name = var_name
		self.permute_data = permute_data

		image_mem = np.array([])

		im_id = 0
		for directory in glob(os.path.join(self.root, "RGBNIRImages", dataset_name, "*")):
			for rgb_filename in glob(os.path.join(directory, "*_RGB.jpg")):
				nir_filename = os.path.join(directory, rgb_filename.split("/")[-1].replace("RGB", "NIR"))
				image_mem = self.read_image(rgb_filename, nir_filename)

				if train_with_patches:
					patches = image_to_patches(image_mem, self.PATCH_SIZE, discard_edges)
					for patch in patches:
						self.images[im_id] = patch
						im_id += 1
					del patches
				else:
					self.images[im_id] = image_mem
					im_id += 1

		im_id = 0
		for directory in glob(os.path.join(self.root, "working_{}".format(dataset_name), "*")):
			for mat_filename in glob(os.path.join(directory, "*.mat")):
				if train_with_patches:
					for i in range(self.IMAGE_SIZE // self.PATCH_SIZE):
						for j in range(self.IMAGE_SIZE // self.PATCH_SIZE):
							self.labels[im_id] = {"mat_path": mat_filename, "idx": (i, j)}
							im_id += 1
				else:
					self.labels[im_id] = mat_filename
					im_id += 1

		if train_with_patches:
			print("Number of RGB Images (Patches): {}".format(len(self.images)))
			print("Number of Hypercubes (Patches): {}".format(len(self.labels)))

		# pair each rgb-nir patch with each hypercube patch
		if permute_data:
			self.permuted_idx = list(itertools.product(self.images.keys(), self.labels.keys()))

	def fetch_image_label(self, index):
		if self.permute_data:
			idx = self.permuted_idx[index]
		else:
			idx = index
		mat_name = self.labels[idx[1]]["mat_path"]
		hypercube = load_mat(mat_name, self.var_name)[self.var_name]
		hypercube = hypercube[:, :, ::BAND_SPACING]
		hypercube = np.transpose(hypercube, [2, 0, 1])

		# getting the desired patch from the hypercube
		i, j = self.labels[idx[1]]["idx"]
		hypercube = hypercube[:, i:i+self.PATCH_SIZE, j:j+self.PATCH_SIZE]
		return self.images[idx[0]], hypercube

	def __len__(self):
		if self.permute_data:
			return len(self.permuted_idx)
		else:
			return len(self.images)

	def __getitem__(self, index):
		image, hypercube = self.fetch_image_label(index)
		return torch.from_numpy(image).float(), torch.from_numpy(hypercube).float()

class DatasetFromDirectory(Dataset):
	def __init__(self, root):
		self.root = root
		self.var_name = var_name
		self.labels = []
		self.images = []

		for filename in glob(os.path.join(self.root, "*_dense_demRGB.png")):
			mat_file_name = filename.split("/")[-1].split("_")[0]
			rgb_img_path = filename
			nir_img_path = os.path.join(self.root, filename.split("/")[-1].replace("RGB", "NIRc"))
			
			rgb = imread(rgb_img_path)
			rgb = rgb/255
			rgb[:,:, [0, 2]] = rgb[:,:, [2, 0]]		# flipping red and blue channels (shape used for training)

			nir = imread(nir_img_path)
			nir = nir/255

			image = np.dstack((rgb, nir))
			image = np.transpose(image, [2, 0, 1])

			self.images.append(image)
			self.labels.append(os.path.join(os.path.dirname(self.root), "mat", mat_file_name + ".mat"))
			del rgb, nir

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		ground_t = load_mat(self.labels[index], self.var_name)
		ground_t = ground_t[self.var_name][:,:,1:204:4] / 4095
		ground_t = np.transpose(ground_t, [2, 0, 1])

		return torch.from_numpy(self.images[index]).float(), torch.from_numpy(ground_t).float()