import os
import itertools
import numpy as np

import h5py
from glob import glob
from imageio import imread
from utils import load_mat

import torch
from torch.utils.data import Dataset
from config import BAND_SPACING, RGBN_BANDS

class DatasetFromHdf5(Dataset):
	def __init__(self, file_path):
		super(DatasetFromHdf5, self).__init__()
		hf = h5py.File(file_path)
		self.images = hf.get("data")
		self.hypercubes = hf.get("label")

	def __getitem__(self, index):
		return torch.from_numpy(self.images[index,:,:,:]).float(), torch.from_numpy(self.hypercubes[index,:,:,:]).float()

	def __len__(self):
		return len(self.images)

def image_to_patches(image, patch_size, discard_edges=True):
	""" 
	Splits the image into patches and returns a list of patches
		image:			RGB-NIR or Hypercube. numpy array of shape (channel, row, col)
		discard_edges:	if True, discard the four corners of the image
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
	rgb[:,:, [0, 2]] = rgb[:,:, [2, 0]]	# flipping red and blue channels (shape used for training)

	nir = imread(nir_filename)[:,:, 0]	# because NIR from the phone is saved as three repeated channels 
	nir = nir/255

	image = np.dstack((rgb, nir))
	image = np.transpose(image, [2, 0, 1])
	del rgb, nir

	return image

def get_normalization_parameters(dataloader):
	""" Give Dataloader and recieve the mean and std of the dataset.
		Note: Make sure that the dataloader is Tensordataset and its not already normalized. """
	channels_sum, channels_squared_sum, num_batches = 0, 0, 0
	for data, _ in dataloader:
		# Mean over batch, height and width, but not over the channels
		channels_sum += torch.mean(data, dim=[0, 2, 3])
		channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
		num_batches += 1

	mean = channels_sum / num_batches

	# std = sqrt(E[X^2] - (E[X])^2)
	std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

	return mean, std

class DatasetFromDirectory(Dataset):
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

	def __init__(self, root, dataset_name=None, product_pairing=True, lazy_read=False, rgbn_from_cube=True, train_with_patches=True, patch_size=64, discard_edges=True):
		"""
		Dataloader for the dataset.
			root:				root directory of the dataset
			dataset_name:		name of the dataset, used to scan over directories (e.g. "oats", "flour", etc.)
			product_pairing:	if True, the each RGB-NIR pair is paired with each hypercube
			lazy_read:			if True, hypercubes are loaded lazily (only when needed)
			rgbn_from_cube:		if True, the RGB-NIR pair is extracted from the hypercube
			train_with_patches:	if True, the RGBN images are split into patches
			patch_size:			size of the patches
			discard_edges:		if True, discard the four corner patches
		"""
		self.PATCH_SIZE = patch_size
		self.root = root
		self.rgbn_from_cube = rgbn_from_cube
		self.lazy_read = lazy_read
		self.product_pairing = product_pairing

		im_id, rgbn_counter = 0, 0
		if not rgbn_from_cube:
			for directory in glob(os.path.join(self.root, "RGBNIRImages", dataset_name, "*")):
				for rgb_filename in glob(os.path.join(directory, "*_RGB.jpg")):
					nir_filename = os.path.join(directory, rgb_filename.split("/")[-1].replace("RGB", "NIR"))
					image = read_image(rgb_filename, nir_filename)
					rgbn_counter += 1

					if train_with_patches:
						patches = image_to_patches(image, self.PATCH_SIZE, discard_edges)
						for patch in patches:
							self.images[im_id] = patch
							im_id += 1
						del patches
					else:
						self.images[im_id] = image
						im_id += 1

		im_id, hypercube_counter = 0, 0
		for directory in glob(os.path.join(self.root, "working_{}".format(dataset_name), "*")):
			for mat_filename in glob(os.path.join(directory, "*.mat")):
				if not lazy_read:
					hypercube = load_mat(mat_filename)
					if rgbn_from_cube:
						image = hypercube[:, :, RGBN_BANDS]
						image = np.transpose(image, [2, 0, 1])
					hypercube = hypercube[:, :, ::BAND_SPACING]
					hypercube = np.transpose(hypercube, [2, 0, 1])

				hypercube_counter += 1
				if train_with_patches:
					for i in range(self.IMAGE_SIZE // self.PATCH_SIZE):
						for j in range(self.IMAGE_SIZE // self.PATCH_SIZE):
							if lazy_read:
								self.labels[im_id] = {"mat_path": mat_filename, "idx": (i, j)}
							else:
								self.labels[im_id] = hypercube[:, i:i+self.PATCH_SIZE, j:j+self.PATCH_SIZE]
								if rgbn_from_cube:
									self.images[im_id] = image[:, i:i+self.PATCH_SIZE, j:j+self.PATCH_SIZE]
							im_id += 1
				else:
					self.labels[im_id] = mat_filename
					im_id += 1

		# pair each rgb-nir patch with each hypercube patch
		if product_pairing:
			self.permuted_idx = list(itertools.product(self.images.keys(), self.labels.keys()))

		print("Number of RGBN Files:\t\t{}\nNumber of Hypercubes:\t\t{}".format(rgbn_counter if not rgbn_from_cube else hypercube_counter, hypercube_counter))
		if train_with_patches:
			print("Number of RGB Images (Patches):\t{}\nNumber of Hypercubes (Patches):\t{}".format(len(self.images), len(self.labels)))

	def fetch_image_label(self, index):
		""" Reads the image and label from the index (lazily or not) and/or product pairs """
		if self.product_pairing:
			idx = self.permuted_idx[index]
		else:
			idx = (index, index)

		if self.lazy_read:
			mat_name = self.labels[idx[1]]["mat_path"]
			hypercube = load_mat(mat_name)
			if self.rgbn_from_cube:
				image = hypercube[:, :, RGBN_BANDS]
				image = np.transpose(image, [2, 0, 1])
			else:
				image = self.images[idx[0]]
			hypercube = hypercube[:, :, ::BAND_SPACING]
			hypercube = np.transpose(hypercube, [2, 0, 1])

			# getting the desired patch from the hypercube
			i, j = self.labels[idx[1]]["idx"]
			hypercube = hypercube[:, i:i+self.PATCH_SIZE, j:j+self.PATCH_SIZE]

			if self.rgbn_from_cube:
				image = image[:, i:i+self.PATCH_SIZE, j:j+self.PATCH_SIZE]
		else:
			image = self.images[idx[0]]
			hypercube = self.labels[idx[1]]

		return image, hypercube

	def __len__(self):
		if self.product_pairing:
			return len(self.permuted_idx)
		else:
			return len(self.images)

	def __getitem__(self, index):
		image, hypercube = self.fetch_image_label(index)
		return torch.from_numpy(image).float(), torch.from_numpy(hypercube).float()