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
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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

def read_image(rgb_filename, nir_filename):
	""" Reads the two images and stack them together while maintaining the order BGR-NIR """
	rgb = imread(rgb_filename)
	rgb[:,:, [0, 2]] = rgb[:,:, [2, 0]]	# flipping red and blue channels (shape used for training)

	nir = imread(nir_filename)
	# because NIR from the phone is saved as three repeated channels
	nir = nir[:,:, 0] if nir.ndim == 3 else np.expand_dims(nir, axis=-1)

	image = np.dstack((rgb, nir))/255.0
	image = np.transpose(image, [2, 0, 1])
	del rgb, nir

	return image

def get_normalization_parameters(dataloader):
	""" Give Dataloader and recieve the mean and std of the dataset.
		Note: Make sure that the dataloader is Tensordataset and its not already normalized. """
	image_channels_sum, image_channels_squared_sum = 0, 0
	label_channels_sum, label_channels_squared_sum, num_batches = 0, 0, 0

	for image, label in dataloader:
		# Mean over batch, height and width, but not over the channels
		image_channels_sum += torch.mean(image, dim=[0, 2, 3])
		image_channels_squared_sum += torch.mean(image**2, dim=[0, 2, 3])

		label_channels_sum += torch.mean(label, dim=[0, 2, 3])
		label_channels_squared_sum += torch.mean(label**2, dim=[0, 2, 3])

		num_batches += 1

	image_mean = image_channels_sum / num_batches
	label_mean = label_channels_sum / num_batches

	# std = sqrt(E[X^2] - (E[X])^2)
	image_std = (image_channels_squared_sum / num_batches - image_mean ** 2) ** 0.5
	label_std = (label_channels_squared_sum / num_batches - label_mean ** 2) ** 0.5

	return (image_mean, image_std), (label_mean, label_std)

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

	def __init__(self, root, dataset_name=None, patch_size=64, lazy_read=False, rgbn_from_cube=True, product_pairing=True, train_with_patches=True, verbose=True, transform=(None, None), value_range=(0, 1)):
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
			verbose:			if True, print the statistics of the dataset
			transform:			two transforms for RGB-NIR and Hypercubes from the dataset
			value_range:		range of the values in the dataset ([0, 1]: MinMaxScaler)
		"""
		self.EPS = 1e-8
		self.root = root
		self.PATCH_SIZE = patch_size
		self.lazy_read = lazy_read
		self.rgbn_from_cube = rgbn_from_cube
		self.product_pairing = product_pairing
		self.input_transform, self.label_transform = transform
		self.value_range = value_range

		im_id, rgbn_counter = 0, 0
		if not rgbn_from_cube:
			for directory in sorted(glob(os.path.join(self.root, "RGBNIRImages", "working_{}".format(dataset_name), "*"))):
				for rgb_filename in sorted(glob(os.path.join(directory, "*_RGB.png"))):
					nir_filename = os.path.join(directory, rgb_filename.split("/")[-1].replace("RGB", "NIR"))
					image = read_image(rgb_filename, nir_filename)
					rgbn_counter += 1

					if train_with_patches:
						for i in range(self.IMAGE_SIZE // self.PATCH_SIZE):
							for j in range(self.IMAGE_SIZE // self.PATCH_SIZE):
								self.images[im_id] = image[:, i*self.PATCH_SIZE:(i+1)*self.PATCH_SIZE, j*self.PATCH_SIZE:(j+1)*self.PATCH_SIZE]
								im_id += 1
					else:
						self.images[im_id] = image
						im_id += 1

		im_id, hypercube_counter = 0, 0
		for directory in sorted(glob(os.path.join(self.root, "working_{}".format(dataset_name), "*"))):
			for mat_filename in sorted(glob(os.path.join(directory, "*.mat"))):
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
								self.labels[im_id] = hypercube[:, i*self.PATCH_SIZE:(i+1)*self.PATCH_SIZE, j*self.PATCH_SIZE:(j+1)*self.PATCH_SIZE]
								if rgbn_from_cube:
									self.images[im_id] = image[:, i*self.PATCH_SIZE:(i+1)*self.PATCH_SIZE, j*self.PATCH_SIZE:(j+1)*self.PATCH_SIZE]
							im_id += 1					
				else:
					self.labels[im_id] = mat_filename
					im_id += 1

		# pair each rgb-nir patch with each hypercube patch
		if product_pairing:
			self.permuted_idx = list(itertools.product(self.images.keys(), self.labels.keys()))
		if verbose:
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
			hypercube = hypercube[:, i*self.PATCH_SIZE:(i+1)*self.PATCH_SIZE, j*self.PATCH_SIZE:(j+1)*self.PATCH_SIZE]

			if self.rgbn_from_cube:
				image = image[:, i*self.PATCH_SIZE:(i+1)*self.PATCH_SIZE, j*self.PATCH_SIZE:(j+1)*self.PATCH_SIZE]
		else:
			image = self.images[idx[0]]
			hypercube = self.labels[idx[1]]

		return image, hypercube

	def __len__(self):
		if self.product_pairing:
			return len(self.permuted_idx)
		else:
			return len(self.images)
		
	def scale_image(self, image, range=(0, 1)):
		""" Scales the image to the desired range """
		dist = image.max(dim=1, keepdim=True)[0] - image.min(dim=1, keepdim=True)[0]
		dist[dist == 0.] = 1.
		scale = 1.0 / dist
		image.mul_(scale).sub_(image.min(dim=1, keepdim=True)[0])
		image.mul_(range[1] - range[0]).add_(range[0])
		return image

	def __getitem__(self, index):
		image, hypercube = self.fetch_image_label(index)
		image = torch.from_numpy(image).float()
		hypercube = torch.from_numpy(hypercube).float()

		if not (self.input_transform == None and self.label_transform == None):
			image = self.input_transform(image)
			hypercube = self.label_transform(hypercube)
		
		if not (self.value_range == None):
			image = self.scale_image(image, self.value_range) + self.EPS
			hypercube = self.scale_image(hypercube, self.value_range) + self.EPS

		# print(image.shape, hypercube.shape)
		# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
		# ax1.imshow(np.transpose(image.numpy(), [1, 2, 0])[:, :, :3])
		# ax1.set_title("Input")
		# ax2.imshow(hypercube[50, :, :].numpy())
		# ax2.set_title("Ground Truth")
		# plt.show()

		# print("Hypercube: %.5f - %.5f, Image: %.5f - %.5f" % (torch.min(hypercube).item(), torch.max(hypercube).item(), torch.min(image).item(), torch.max(image).item()))
		# print(torch.any(torch.isnan(hypercube)), torch.any(torch.isnan(image)))

		return image, hypercube