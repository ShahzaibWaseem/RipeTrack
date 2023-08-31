import os
import random
import itertools
import numpy as np

import h5py
from glob import glob

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

from utils import load_mat, read_image, data_augmentation, get_normalization_parameters, visualize_data_item
from config import BAND_SPACING, RGBN_BANDS, BANDS, TEST_DATASETS, GT_RGBN_DIR_NAME, TRAIN_DATASET_DIR, TRAIN_DATASET_FILES, VALID_DATASET_FILES, TEST_ROOT_DATASET_DIR, APPLICATION_NAME, PATCH_SIZE, RECONSTRUCTED_HS_DIR_NAME, batch_size, device

def get_dataloaders(input_transform, label_transform, task, load_from_h5=False, trainset_size=0.7):
	if load_from_h5:
		train_data, valid_data = [], []
		for datasetFile in TRAIN_DATASET_FILES:
			h5_filepath = os.path.join(TRAIN_DATASET_DIR, datasetFile)
			dataset = DatasetFromHdf5(h5_filepath)
			train_data.append(dataset)
			print("Length of Training Set (%s):\t" % datasetFile, len(dataset))

		for datasetFile in VALID_DATASET_FILES:
			h5_filepath = os.path.join(TRAIN_DATASET_DIR, datasetFile)
			dataset = DatasetFromHdf5(h5_filepath)
			valid_data.append(dataset)
			print("Length of Validation Set (%s):\t" % datasetFile, len(dataset))
	else:
		dataset = DatasetFromDirectory(root=TEST_ROOT_DATASET_DIR,
									   application_name=APPLICATION_NAME,
									   task=task,
									   patch_size=PATCH_SIZE,
									   lazy_read=True,
									   shuffle=True,
									   rgbn_from_cube=False,
									   use_all_bands=False if task == "reconstruction" else True,
									   train_with_patches=True,
									   positive_only=True,
									   verbose=True,
									   augment_factor=0 if task == "reconstruction" else 8,
									   transform=(input_transform, label_transform))

		test_data_loader = DataLoader(dataset=dataset,
									  num_workers=0,
									  batch_size=1,
									  shuffle=False,
									  pin_memory=False) if task == "classification" else dataset

		train_data, valid_data = random_split(dataset, [int(trainset_size*len(dataset)), len(dataset) - int(len(dataset)*trainset_size)])

		print("Length of Training Set ({}%):\t\t{}".format(round(trainset_size * 100), len(train_data)))
		print("Length of Validation Set ({}%):\t\t{}".format(round((1-trainset_size) * 100), len(valid_data)))
	train_data_loader = DataLoader(dataset=train_data,
								   num_workers=2,
								   batch_size=batch_size,
								   shuffle=True,
								   pin_memory=True)

	valid_data_loader = DataLoader(dataset=valid_data,
								   num_workers=2,
								   batch_size=4,
								   shuffle=False,
								   pin_memory=True)

	return train_data_loader, valid_data_loader, test_data_loader

def get_required_transforms(task="reconstruction"):
	""" Returns the two transforms for the RGB-NIR image input and Hypercube label.
		Note: The `dataset` recieved is already Tensor data.
		This function gets the dataset specified in the `config.py` file. """
	# Dataset
	image_mean, image_std, hypercube_mean, hypercube_std = 0, 0, 0, 0

	dataset = DatasetFromDirectory(root=TEST_ROOT_DATASET_DIR,
								   application_name=APPLICATION_NAME,
								   task=task,
								   patch_size=PATCH_SIZE,
								   lazy_read=False,
								   shuffle=False,
								   rgbn_from_cube=False,
								   use_all_bands=False if task == "reconstruction" else True,
								   train_with_patches=True,
								   positive_only=False,
								   verbose=False,
								   augment_factor=0,
								   transform=(None, None))

	dataloader = DataLoader(dataset=dataset,
							num_workers=0,
							batch_size=batch_size,
							shuffle=False,
							pin_memory=False)

	(image_mean, image_std), (hypercube_mean, hypercube_std) = get_normalization_parameters(dataloader)

	print(75*"-" + "\nDataset Normalization\n" + 75*"-")

	if task == "reconstruction":
		print("RGB-NIR Images Size:\t\t\t\t\t%d" % (image_mean.size(dim=0)))
		print("The Mean of the dataset is in the range:\t\t%f - %f"
			% (torch.min(image_mean).item(), torch.max(image_mean).item()))
		print("The Standard Deviation of the dataset is in the range:\t%f - %f\n"
			% (torch.min(image_std).item(), torch.max(image_std).item()))

	print("Hypercubes Size:\t\t\t\t\t%d" % (hypercube_std.size(dim=0)))
	print("The Mean of the dataset is in the range:\t\t%f - %f"
		  % (torch.min(hypercube_mean).item(), torch.max(hypercube_mean).item()))
	print("The Standard Deviation of the dataset is in the range:\t%f - %f"
		  % (torch.min(hypercube_std).item(), torch.max(hypercube_std).item()))
	print(75*"-")
	del dataset, dataloader

	hypercube_transform_list = []

	if task == "classification":
		hypercube_transform_list = [# transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
									# transforms.RandomResizedCrop(size=(PATCH_SIZE, PATCH_SIZE)),
									# transforms.CenterCrop(size=(PATCH_SIZE, PATCH_SIZE)),
									transforms.RandomRotation(20, interpolation=transforms.InterpolationMode.BILINEAR),
									transforms.RandomHorizontalFlip()]

	# Data is already tensor, so just normalize it
	hypercube_transform_list.append(transforms.Normalize(mean=hypercube_mean, std=hypercube_std))
	input_transform = transforms.Compose([transforms.Normalize(mean=image_mean, std=image_std)]) if task == "reconstruction" else None
	hypercube_transform = transforms.Compose([transforms.Normalize(mean=hypercube_mean, std=hypercube_std)])

	del hypercube_transform_list
	return input_transform, hypercube_transform

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

class DatasetFromDirectory(Dataset):
	# Expects the directory structure to be:
	# root/
	# 	application_name/		(shelflife)
	#		DatasetName/			(avocado-emp_204ch - Hypercubes)
	#			rgbn/				(RGB-NIR Images)
	#				01_[RGB/NIR].png
	#				...
	# 			mobile-rgbn/		(RGB-NIR Images captured by the mobile phone)
	#				01_[RGB/NIR].png
	#				...
	#			secondary-rgbn/		(RGB-NIR Images from the Hyperspectral Camera's Secondary Camera - RGB)
	#				01_[RGB/NIR].png
	#				...
	#			01.mat
	#			...
	#		DatasetName/			(pear-williams_204ch - Hypercubes)
	#				...

	EPS = 1e-8
	IMAGE_SIZE = 512
	images, hypercubes, classlabels, actual_classlabels = {}, {}, {}, {}
	min_values = (torch.tensor([float("Inf"), float("Inf")]))		# Image Min: -1.30753493309021, Hypercube Min: -2.123685598373413
	max_values = (torch.tensor([float("-Inf"), float("-Inf")]))		# Image Max: -1.30753493309021, Hypercube Min: -2.123685598373413

	def __init__(self, root, application_name=None, task="reconstruction", patch_size=64, lazy_read=False, shuffle=True, rgbn_from_cube=True, use_all_bands=True, train_with_patches=True, positive_only=True, crop_size=0, augment_factor=8, verbose=True, transform=(None, None)):
		"""
		Dataloader for the dataset.
			root:				root directory of the dataset
			application_name:	name of the application, contains the datasets (e.g. "shelflife", etc.)
			task:				Deep Learning tasks. "reconstruction" or "classification"
			patch_size:			size of the patches
			lazy_read:			if True, hypercubes are loaded lazily (only when needed)
			rgbn_from_cube:		if True, the RGB-NIR pair is extracted from the hypercube
			use_all_bands:		if True, use all, 204, bands (ideal case is for reconstruction)
			train_with_patches:	if True, the RGBN images are split into patches
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
		self.positive_only = positive_only
		self.input_transform, self.label_transform = transform
		self.crop_size = crop_size
		self.augment_factor = augment_factor
		self.train_with_patches = self.PATCH_SIZE > 0
		self.verbose = verbose

		self.IMAGE_SIZE -= self.crop_size if self.crop_size > 0 else 0

		number_of_files = len([hypercube for directory in glob(os.path.join(self.root, application_name, "*")) for hypercube in glob(os.path.join(directory, "*.mat"))])
		number_of_files *= self.augment_factor if self.augment_factor > 0 else 1

		self.augment_factor = self.augment_factor if self.augment_factor > 0 else 1			# just to make sure script runs even when augment_factor is 0 (otherwise for)

		if train_with_patches:
			number_of_files = number_of_files * ((self.IMAGE_SIZE // self.PATCH_SIZE) ** 2)
			if shuffle:
				self.idxlist = random.sample(range(number_of_files), number_of_files)
			else:
				self.idxlist = list(range(number_of_files))
		
		print("Shuffle IDX:", self.idxlist[:10]) if self.verbose else None

		global BAND_SPACING, BANDS
		# BAND_SPACING = 1 if task == "classification" else 4

		print("Reading RGBN Images from:") if self.verbose else None

		im_id, rgbn_counter = 0, 0
		shelf_life_root_directory = os.path.join(self.root, application_name)
		if not rgbn_from_cube:
			for dataset in TEST_DATASETS:
				directory = os.path.join(shelf_life_root_directory, "{}_204ch".format(dataset), GT_RGBN_DIR_NAME)
				print(" " * 25, directory) if self.verbose else None
				for rgb_filename in sorted(glob(os.path.join(directory, "*_RGB.png"))):
					classlabel = directory.split("/")[-2].split("_")[0]
					actual_classlabel = classlabel
					nir_filename = os.path.join(directory, rgb_filename.split("/")[-1].replace("RGB", "NIR"))
					orig_image = read_image(rgb_filename, nir_filename)
					# image = crop_image(image, start=self.crop_size, end=self.IMAGE_SIZE-self.crop_size) if task == "classification" and self.crop_size>0 else image
					for aug_mode in range(self.augment_factor):			# Augment the dataset
						image = data_augmentation(orig_image, aug_mode) if task == "classification" else orig_image
						image = np.transpose(image, [2, 0, 1])
						image = torch.from_numpy(image.copy()).float()
						image = self.input_transform(image) if not self.input_transform == None else image
						rgbn_counter += 1

						if train_with_patches:
							for i in range(image.size(1) // self.PATCH_SIZE):
								for j in range(image.size(2) // self.PATCH_SIZE):
									self.images[self.idxlist[im_id]] = image[:, i*self.PATCH_SIZE:(i+1)*self.PATCH_SIZE, j*self.PATCH_SIZE:(j+1)*self.PATCH_SIZE]
									self.classlabels[self.idxlist[im_id]] = np.long(TEST_DATASETS.index(classlabel))
									self.actual_classlabels[self.idxlist[im_id]] = np.long(TEST_DATASETS.index(actual_classlabel))
									im_id += 1
						else:
							self.images[self.idxlist[im_id]] = image
							self.classlabels[self.idxlist[im_id]] = np.long(TEST_DATASETS.index(classlabel))
							self.actual_classlabels[self.idxlist[im_id]] = np.long(TEST_DATASETS.index(actual_classlabel))
							im_id += 1

		print("\nReading Hyper cubes from:") if self.verbose else None

		im_id, hypercube_counter = 0, 0
		for dataset in TEST_DATASETS:
			directory = os.path.join(shelf_life_root_directory, "{}_204ch".format(dataset))
			directory = os.path.join(directory, RECONSTRUCTED_HS_DIR_NAME) if task == "classification" else directory
			print(" " * 25 + directory) if self.verbose else None
			for mat_filename in sorted(glob(os.path.join(directory, "*.mat"))):
				for aug_mode in range(self.augment_factor):				# Augment the dataset
					if not lazy_read:
						hypercube = load_mat(mat_filename)
						hypercube = hypercube[:, :, BANDS] if not self.use_all_bands else hypercube
						hypercube = data_augmentation(hypercube, aug_mode)
						hypercube = np.transpose(hypercube, [2, 0, 1])
						# hypercube = crop_image(hypercube, start=self.crop_size, end=self.IMAGE_SIZE-self.crop_size) if task == "classification" and self.crop_size>0 else hypercube
						hypercube = torch.from_numpy(hypercube.copy()).float()

						if rgbn_from_cube:
							image = hypercube[RGBN_BANDS, :, :]
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

		assert len(self.images) == len(self.hypercubes) == len(self.classlabels) == len(self.actual_classlabels), "Number of images and hypercubes and classlabels do not match."
		if verbose:
			print("\nBands used:\t\t\t\t{}\nBand Numbers:\t\t\t\t{}".format(BANDS if not self.use_all_bands else "Using the reconstructed images, check config.py" if task == "classification" else list(range(204)), len(BANDS)))
			# print("Number of RGBN Files:\t\t\t{}\nNumber of Hypercubes Files:\t\t{}".format(rgbn_counter//augment_factor if not rgbn_from_cube else hypercube_counter, hypercube_counter//augment_factor))
			print("Augmentation factor:\t\t\t{}".format(self.augment_factor))
			print("Number of RGBN Images:\t\t\t{}\nNumber of Hypercubes Images:\t\t{}".format(rgbn_counter if not rgbn_from_cube else hypercube_counter, hypercube_counter))

			if train_with_patches and PATCH_SIZE != self.IMAGE_SIZE:
				print("Patch Size:\t\t\t\t{}\nNumber of Patches ({}/{} * {}):\t{}".format(self.PATCH_SIZE, self.IMAGE_SIZE, self.PATCH_SIZE, rgbn_counter, len(self.hypercubes)))

			print("Images Shape:\t\t\t\t{}\nHypercubes Shape:\t\t\t{}".format(list(self.images[0].size()), list(self.hypercubes[0].size()) if not lazy_read else "Lazy Read. Size decided later."))

	def fetch_image_label(self, index):
		""" Reads the image and label from the index (lazily or not) and/or product pairs """
		idx = (index, index)	

		if self.lazy_read:
			mat_name = self.hypercubes[idx[1]]["mat_path"]
			aug_mode = self.hypercubes[idx[1]]["aug_mode"]
			hypercube = load_mat(mat_name)
			hypercube = hypercube[:, :, BANDS] if not self.use_all_bands else hypercube
			hypercube = data_augmentation(hypercube, aug_mode) if self.task == "classification" else hypercube
			hypercube = np.transpose(hypercube, [2, 0, 1])
			hypercube = torch.from_numpy(hypercube.copy()).float()
			classlabel = self.classlabels[idx[0]]
			actual_classlabel = self.actual_classlabels[idx[0]]

			if self.rgbn_from_cube:
				image = hypercube[RGBN_BANDS, :, :]
			else:
				image = self.images[idx[0]]
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
			actual_classlabel = self.actual_classlabels[idx[0]]

		return image, hypercube, classlabel, actual_classlabel

	def __len__(self):
		return len(self.images)
		
	def __getitem__(self, index):
		image, hypercube, classlabel, actual_classlabel = self.fetch_image_label(index)

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

		return image, hypercube, classlabel, actual_classlabel