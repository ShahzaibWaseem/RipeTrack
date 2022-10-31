import os
import random
import itertools
import numpy as np

import h5py
from glob import glob
from imageio import imread
from utils import load_mat

import torch
from torch.utils.data import Dataset
from config import BAND_SPACING, RGBN_BANDS, BANDS, TEST_DATASETS

import matplotlib.pyplot as plt

class DatasetFromHdf5(Dataset):
	def __init__(self, file_path):
		super(DatasetFromHdf5, self).__init__()
		hdf5_file = h5py.File(file_path)
		self.images = hdf5_file.get("data")
		self.hypercubes = hdf5_file.get("label")

	def __getitem__(self, index):
		return torch.from_numpy(self.images[index, :, :, :]).float(), torch.from_numpy(self.hypercubes[index, :, :, :]).float()

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
	del rgb, nir

	return image

def scale_image(image, range=(0, 1)):
	""" Scales the image to the desired range.
		Depreciated: Will be removed in the future. """
	dist = image.max(dim=1, keepdim=True)[0] - image.min(dim=1, keepdim=True)[0]
	dist[dist == 0.] = 1.
	scale = 1.0 / dist
	image.mul_(scale).sub_(image.min(dim=1, keepdim=True)[0])
	image.mul_(range[1] - range[0]).add_(range[0])
	return image

def get_normalization_parameters(dataloader):
	""" Give Dataloader and recieve the mean and std of the dataset.
		Note: Make sure that the dataloader is Tensordataset and its not already normalized. """
	image_channels_sum, image_channels_squared_sum = 0, 0
	hypercube_channels_sum, hypercube_channels_squared_sum, num_batches = 0, 0, 0

	for image, hypercube, _ in dataloader:
		# Mean over batch, height and width, but not over the channels
		image_channels_sum += torch.mean(image, dim=[0, 2, 3])
		image_channels_squared_sum += torch.mean(image**2, dim=[0, 2, 3])

		hypercube_channels_sum += torch.mean(hypercube, dim=[0, 2, 3])
		hypercube_channels_squared_sum += torch.mean(hypercube**2, dim=[0, 2, 3])

		num_batches += 1
	
	print("number of batches", num_batches)

	image_mean = image_channels_sum / num_batches
	hypercube_mean = hypercube_channels_sum / num_batches

	# std = sqrt(E[X^2] - (E[X])^2)
	image_std = (image_channels_squared_sum / num_batches - image_mean ** 2) ** 0.5
	hypercube_std = (hypercube_channels_squared_sum / num_batches - hypercube_mean ** 2) ** 0.5

	return (image_mean, image_std), (hypercube_mean, hypercube_std)

def crop_image(image, start, end):
	""" Crops the image to the desired range. 
		Note: This function expects the image to be in the format [C, H, W] and H = W. """
	return image[:, start:end, start:end]

def data_augmentation(image, aug_mode=0):
	if aug_mode == 0:
		return image								# original image
	elif aug_mode == 1:
		return np.flipud(image)						# flip up and down
	elif aug_mode == 2:
		return np.rot90(image)						# rotate counterwise 90 degree
	elif aug_mode == 3:
		return np.flipud(np.rot90(image))			# rotate 90 degree and flip up and down
	elif aug_mode == 4:
		return np.rot90(image, k=2)					# rotate 180 degree
	elif aug_mode == 5:
		return np.flipud(np.rot90(image, k=2))		# rotate 180 degree and flip
	elif aug_mode == 6:
		return np.rot90(image, k=3)					# rotate 270 degree
	elif aug_mode == 7:
		return np.flipud(np.rot90(image, k=3))		# rotate 270 degree and flip
	else:
		return

def visualize_data_item(image, hypercube, band, classlabel):
	fig, ax = plt.subplots(1, 2, figsize=(10, 5))
	fig.suptitle("Class: %s" % TEST_DATASETS[classlabel])

	# visualizing it in RGB (instead of BGR)
	image=np.transpose(image.numpy()[:3], (1, 2, 0))[:,:, [2, 1, 0]]
	ax[0].imshow(image)
	ax[0].set_xlabel(image.shape)
	ax[0].set_title("RGBN - 0:3 (RGB)")
	ax[1].imshow(hypercube.numpy()[band])
	ax[1].set_xlabel(hypercube.numpy().shape)
	ax[1].set_title("Hypercube - %i" % band)
	plt.show()

class DatasetFromDirectory(Dataset):
	# Expects the directory structure to be:
	# root/
	#	RGBNIRImages/
	#		DatasetName			(working_organic)
	#			label1/			(working_gala-organic etc)
	#				01_RGB.jpg
	#				01_NIR.jpg
	#				...
	#			label2/
	#				...
	#	DatasetName/			(working_organic - Hypercubes)
	#		label1/				(working_ambrosia-nonorganic_204ch etc)
	#			01.mat
	#			...
	#		label2/
	#			...

	EPS = 1e-8
	IMAGE_SIZE = 512
	images, hypercubes, classlabels = {}, {}, {}
	min_values = (torch.tensor([float("Inf"), float("Inf")]))		# Image Min: -1.30753493309021, Hypercube Min: -2.123685598373413
	max_values = (torch.tensor([float("-Inf"), float("-Inf")]))		# Image Max: -1.30753493309021, Hypercube Min: -2.123685598373413

	def __init__(self, root, dataset_name=None, task="reconstruction", patch_size=64, lazy_read=False, shuffle=True, rgbn_from_cube=True, use_all_bands=True, product_pairing=False, train_with_patches=True, positive_only=True, crop_size=0, augment_factor=8, verbose=True, transform=(None, None)):
		"""
		Dataloader for the dataset.
			root:				root directory of the dataset
			dataset_name:		name of the dataset, used to scan over directories (e.g. "oats", "flour", etc.)
			task:				Deep Learning tasks. "reconstruction" or "classification"
			patch_size:			size of the patches
			lazy_read:			if True, hypercubes are loaded lazily (only when needed)
			rgbn_from_cube:		if True, the RGB-NIR pair is extracted from the hypercube
			use_all_bands:		if True, use all, 204, bands (ideal case is for reconstruction)
			product_pairing:	if True, the each RGB-NIR pair is paired with each hypercube						(Will be deprecated)
			train_with_patches:	if True, the RGBN images are split into patches
			discard_edges:		if True, discard the four corner patches
			positive_only:		if True, make both the images and hypercubes positive
			verbose:			if True, print the statistics of the dataset
			transform:			two transforms for RGB-NIR and Hypercubes from the dataset
		"""
		self.root = root
		self.task = task
		self.PATCH_SIZE = patch_size
		self.lazy_read = lazy_read
		self.rgbn_from_cube = rgbn_from_cube
		self.use_all_bands = use_all_bands
		self.product_pairing = product_pairing
		self.positive_only = positive_only
		self.input_transform, self.label_transform = transform
		self.crop_size = crop_size
		self.augment_factor = augment_factor
		self.train_with_patches = self.PATCH_SIZE > 0
		self.verbose = verbose

		self.IMAGE_SIZE -= self.crop_size if self.crop_size > 0 else 0

		number_of_files = len([hypercube for directory in glob(os.path.join(self.root, "working_{}".format(dataset_name), "*")) for hypercube in glob(os.path.join(directory, "*.mat"))])
		number_of_files *= self.augment_factor if self.augment_factor > 0 else 1

		self.augment_factor = self.augment_factor if self.augment_factor > 0 else 1			# just to make sure script runs even when augment_factor is 0 (otherwise for)

		if train_with_patches:
			number_of_files = number_of_files * ((self.IMAGE_SIZE // self.PATCH_SIZE) ** 2)
			if shuffle:
				self.idxlist = random.sample(range(number_of_files), number_of_files)
			else:
				self.idxlist = list(range(number_of_files))

		global BAND_SPACING, BANDS
		# BAND_SPACING = 1 if task == "classification" else 4

		print("Reading RGBN Images from:") if self.verbose else None

		im_id, rgbn_counter = 0, 0
		if not rgbn_from_cube:
			for directory in sorted(glob(os.path.join(self.root, "RGBNIRImages", "working_{}".format(dataset_name), "*"))):
				print(" " * 25 + directory) if self.verbose else None
				for rgb_filename in sorted(glob(os.path.join(directory, "*_RGB.png"))):
					classlabel = directory.split("/")[-1].split("_")[1]
					classlabel = classlabel.split("-")
					classlabel = classlabel[0] if len(classlabel) == 1 else classlabel[1]
					nir_filename = os.path.join(directory, rgb_filename.split("/")[-1].replace("RGB", "NIR"))
					orig_image = read_image(rgb_filename, nir_filename)
					# image = crop_image(image, start=self.crop_size, end=self.IMAGE_SIZE-self.crop_size) if task == "classification" and self.crop_size>0 else image
					for aug_mode in range(self.augment_factor):			# Augment the dataset
						image = data_augmentation(orig_image, aug_mode)
						image = np.transpose(image, [2, 0, 1])
						image = torch.from_numpy(image.copy()).float()
						image = self.input_transform(image) if not self.input_transform == None else image
						rgbn_counter += 1

						if train_with_patches:
							for i in range(image.size(1) // self.PATCH_SIZE):
								for j in range(image.size(2) // self.PATCH_SIZE):
									self.images[self.idxlist[im_id]] = image[:, i*self.PATCH_SIZE:(i+1)*self.PATCH_SIZE, j*self.PATCH_SIZE:(j+1)*self.PATCH_SIZE]
									self.classlabels[self.idxlist[im_id]] = np.long(TEST_DATASETS.index(classlabel))
									im_id += 1
						else:
							self.images[self.idxlist[im_id]] = image
							self.classlabels[self.idxlist[im_id]] = np.long(TEST_DATASETS.index(classlabel))
							im_id += 1

		print("\nReading Hyper cubes from:") if self.verbose else None

		im_id, hypercube_counter = 0, 0
		for directory in sorted(glob(os.path.join(self.root, "working_{}".format(dataset_name), "*"))):
			directory = os.path.join(directory, "inference") if task == "classification" else directory
			print(" " * 25 + directory) if self.verbose else None
			for mat_filename in sorted(glob(os.path.join(directory, "*.mat"))):
				for aug_mode in range(self.augment_factor):				# Augment the dataset
					if not lazy_read:
						hypercube = load_mat(mat_filename)
						hypercube = data_augmentation(hypercube, aug_mode)
						hypercube = np.transpose(hypercube, [2, 0, 1])
						# hypercube = crop_image(hypercube, start=self.crop_size, end=self.IMAGE_SIZE-self.crop_size) if task == "classification" and self.crop_size>0 else hypercube
						hypercube = torch.from_numpy(hypercube.copy()).float()

						if rgbn_from_cube:
							image = hypercube[RGBN_BANDS, :, :]
						hypercube = hypercube[BANDS, :, :] if not self.use_all_bands else hypercube
						hypercube = self.label_transform(hypercube) if not self.label_transform == None else hypercube

					hypercube_counter += 1
					if train_with_patches:
						for i in range(self.IMAGE_SIZE // self.PATCH_SIZE):
							for j in range(self.IMAGE_SIZE // self.PATCH_SIZE):
								if lazy_read:
									self.hypercubes[self.idxlist[im_id]] = {"mat_path": mat_filename, "idx": (i, j), "aug_mode": aug_mode}
								else:
									self.hypercubes[self.idxlist[im_id]] = hypercube[:, i*self.PATCH_SIZE:(i+1)*self.PATCH_SIZE, j*self.PATCH_SIZE:(j+1)*self.PATCH_SIZE]
									if rgbn_from_cube:
										self.images[self.idxlist[im_id]] = image[:, i*self.PATCH_SIZE:(i+1)*self.PATCH_SIZE, j*self.PATCH_SIZE:(j+1)*self.PATCH_SIZE]
								im_id += 1
					else:
						self.hypercubes[im_id] = hypercube
						im_id += 1

		assert len(self.images) == len(self.hypercubes) == len(self.classlabels), "Number of images and hypercubes and classlabels do not match."
		# pair each rgb-nir patch with each hypercube patch
		if product_pairing:
			self.permuted_idx = list(itertools.product(self.images.keys(), self.hypercubes.keys()))
		if verbose:
			print("BANDS used:", BANDS if not self.use_all_bands else list(range(204)))
			# print("Shuffled Indices:", self.idxlist)
			print("Number of RGBN Files:\t\t\t{}\nNumber of Hypercubes:\t\t\t{}".format(rgbn_counter if not rgbn_from_cube else hypercube_counter, hypercube_counter))

			if train_with_patches:
				print("Patch Size:\t\t\t\t{}\nNumber of Patches ({}/{} * {}):\t{}".format(self.PATCH_SIZE, self.IMAGE_SIZE, self.PATCH_SIZE, rgbn_counter, len(self.hypercubes)))

			print("Images Shape:\t\t\t\t{}\nHypercubes Shape:\t\t\t{}".format(list(self.images[0].size()), list(self.hypercubes[0].size())))

	def fetch_image_label(self, index):
		""" Reads the image and label from the index (lazily or not) and/or product pairs """
		if self.product_pairing:
			idx = self.permuted_idx[index]
		else:
			idx = (index, index)

		if self.lazy_read:
			mat_name = self.hypercubes[idx[1]]["mat_path"]
			aug_mode = self.hypercubes[idx[1]]["aug_mode"]
			hypercube = load_mat(mat_name)
			hypercube = data_augmentation(hypercube, aug_mode)
			hypercube = np.transpose(hypercube, [2, 0, 1])
			hypercube = torch.from_numpy(hypercube.copy()).float()
			classlabel = self.classlabels[idx[0]]

			if self.rgbn_from_cube:
				image = hypercube[RGBN_BANDS, :, :]
			else:
				image = self.images[idx[0]]
			hypercube = hypercube[BANDS, :, :] if not self.use_all_bands else hypercube
			hypercube = self.label_transform(hypercube) if not self.label_transform == None else hypercube

			# getting the desired patch from the hypercube
			i, j = self.hypercubes[idx[1]]["idx"]
			hypercube = hypercube[:, i*self.PATCH_SIZE:(i+1)*self.PATCH_SIZE, j*self.PATCH_SIZE:(j+1)*self.PATCH_SIZE]

			if self.rgbn_from_cube:
				image = image[:, i*self.PATCH_SIZE:(i+1)*self.PATCH_SIZE, j*self.PATCH_SIZE:(j+1)*self.PATCH_SIZE]
		else:
			image = self.images[idx[0]]
			hypercube = self.hypercubes[idx[1]]
			classlabel = self.classlabels[idx[0]]

		return image, hypercube, classlabel

	def __len__(self):
		if self.product_pairing:
			return len(self.permuted_idx)
		else:
			return len(self.images)
		
	def __getitem__(self, index):
		image, hypercube, classlabel = self.fetch_image_label(index)

		# self.min_values = (torch.min(torch.tensor([torch.min(image), self.min_values[0]])).item(), torch.min(torch.tensor([torch.min(hypercube), self.min_values[1]])).item())
		# self.max_values = (torch.max(torch.tensor([torch.max(image), self.max_values[0]])).item(), torch.max(torch.tensor([torch.max(hypercube), self.max_values[1]])).item())

		# if self.verbose:
		# 	print("Image: %+.5f, %+.5f, Hypercube: %+.5f, %+.5f\t\tMin Values: %s\t Max Values: %s" % \
		# 		(torch.min(image).item(), torch.max(image).item(), torch.min(hypercube).item(), torch.max(hypercube).item(), self.min_values, self.max_values))

		# visualize_data_item(image, hypercube, 12, classlabel)

		if self.positive_only:
			# Image Min: -1.1055755615234375, Hypercube Min: -1.3652015924453735, Inference Min: -1.7359950542449951
			# 0.1 is to compensate for conversion errors
			image = image + 1.1055755615234375 + 0.1
			hypercube = hypercube + 1.3652015924453735 if self.task == "reconstruction" else hypercube + 1.7359950542449951
			hypercube = hypercube + 0.1

		return image, hypercube, classlabel