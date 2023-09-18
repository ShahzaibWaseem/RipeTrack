import os
import random
import numpy as np

import h5py
from glob import glob
from imageio import imread

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

from utils import load_mat, read_image, data_augmentation, get_normalization_parameters, visualize_data_item
from config import APPEND_SECONDARY_RGB_CAM_INPUT, GT_SECONDARY_RGB_CAM_DIR_NAME, BAND_SPACING, RGBN_BANDS, BANDS, TEST_DATASETS, GT_RGBN_DIR_NAME, TRAIN_DATASET_DIR, TRAIN_DATASET_FILES, VALID_DATASET_FILES, TEST_ROOT_DATASET_DIR, APPLICATION_NAME, PATCH_SIZE, RECONSTRUCTED_HS_DIR_NAME, EPS, batch_size, device

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
									   append_secondary_input=APPEND_SECONDARY_RGB_CAM_INPUT,
									   augment_factor=0 if task == "reconstruction" else 8,
									   transform=(input_transform, label_transform))

		test_data_loader = DataLoader(dataset=dataset,
									  num_workers=4,
									  batch_size=1,
									  shuffle=False,
									  pin_memory=False) if task == "classification" else dataset

		train_data, valid_data = random_split(dataset, [int(trainset_size*len(dataset)), len(dataset) - int(len(dataset)*trainset_size)])

		print("Length of Training Set ({}%):\t\t{}".format(round(trainset_size * 100), len(train_data)))
		print("Length of Validation Set ({}%):\t\t{}".format(round((1-trainset_size) * 100), len(valid_data)))
	train_data_loader = DataLoader(dataset=train_data,
								   num_workers=4,
								   batch_size=batch_size,
								   shuffle=True,
								   pin_memory=False)

	valid_data_loader = DataLoader(dataset=valid_data,
								   num_workers=4,
								   batch_size=4,
								   shuffle=False,
								   pin_memory=False)

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
								   verbose=True,
								   append_secondary_input=APPEND_SECONDARY_RGB_CAM_INPUT,
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

	def __init__(self, root, application_name=None, task="reconstruction", patch_size=64, lazy_read=False, shuffle=True, rgbn_from_cube=True, use_all_bands=True, train_with_patches=True, positive_only=True, crop_size=0, append_secondary_input=False, augment_factor=8, verbose=True, transform=(None, None)):
		"""
		Dataloader for the dataset.
			root:					root directory of the dataset
			application_name:		name of the application, contains the datasets (e.g. "shelflife", etc.)
			task:					Deep Learning tasks. "reconstruction" or "classification"
			patch_size:				size of the patches
			lazy_read:				if True, hypercubes are loaded lazily (only when needed)
			rgbn_from_cube:			if True, the RGB-NIR pair is extracted from the hypercube
			use_all_bands:			if True, use all, 204, bands (ideal case is for reconstruction)
			train_with_patches:		if True, the RGBN images are split into patches
			positive_only:			if True, make both the images and hypercubes positive
			crop_size:				size of the crop
			append_secondary_input:	if True, append the secondary RGB-NIR images (from the secondary RGB camera on HS Cam) to the inputs
			augment_factor:			number of times to augment the dataset
			verbose:				if True, print the statistics of the dataset
			transform:				two transforms for RGB-NIR and Hypercubes from the dataset
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

		self.augment_factor = self.augment_factor if self.augment_factor > 0 else 1			# just to make sure script runs even when augment_factor is 0 (otherwise for)

		number_of_files = len([hypercube for directory in glob(os.path.join(self.root, application_name, "*")) for hypercube in glob(os.path.join(directory, "*.mat"))])
		number_of_files *= self.augment_factor

		if append_secondary_input:
			number_of_files *= 2

		if train_with_patches:
			number_of_files = number_of_files * ((self.IMAGE_SIZE // self.PATCH_SIZE) ** 2)
			if shuffle:
				self.idxlist = random.sample(range(number_of_files), number_of_files)
			else:
				self.idxlist = list(range(number_of_files))

		global BAND_SPACING, BANDS
		# BAND_SPACING = 1 if task == "classification" else 4

		directories_considered = [GT_RGBN_DIR_NAME]
		if append_secondary_input:
			directories_considered.append(GT_SECONDARY_RGB_CAM_DIR_NAME)

		print("Reading RGBN Images from:") if self.verbose else None

		im_id, rgbn_counter = 0, 0
		shelf_life_root_directory = os.path.join(self.root, application_name)
		if not rgbn_from_cube:
			for dataset in TEST_DATASETS:
				directory = os.path.join(shelf_life_root_directory, "{}_204ch".format(dataset))
				print(" " * 25, directory, "\t{} and {}".format(GT_RGBN_DIR_NAME, GT_SECONDARY_RGB_CAM_DIR_NAME) if append_secondary_input else GT_RGBN_DIR_NAME) if self.verbose else None
				for append_dir in directories_considered:
					for rgb_filename in sorted(glob(os.path.join(directory, append_dir, "*_RGB.png"))):
						classlabel = os.path.split(directory)[-1].split("_")[0]
						actual_classlabel = classlabel
						nir_filename = os.path.join(directory, append_dir, os.path.split(rgb_filename)[-1].replace("RGB", "NIR"))
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
			for run in range(2 if append_secondary_input else 1):
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
			print("\nBands used:\t\t\t\t{}\nNumber of Bands:\t\t\t{}".format(BANDS if not self.use_all_bands else "Using the reconstructed images, check config.py" if task == "classification" else list(range(204)), len(BANDS)))
			print("Shuffle Indices:\t\t\t{}".format(self.idxlist[:10]))
			print("Number of RGBN Files:\t\t\t{}\nNumber of Hypercubes Files:\t\t{}".format((rgbn_counter//len(directories_considered))//self.augment_factor if not rgbn_from_cube else hypercube_counter, (hypercube_counter//len(directories_considered))//self.augment_factor))
			print("Augmentation factor:\t\t\t{}".format(self.augment_factor))
			print("Appended Secondary Input:\t\t{}".format("yes" if append_secondary_input else "no"))
			print("Number of RGBN Images:\t\t\t{}\nNumber of Hypercubes Images:\t\t{}".format(rgbn_counter if not rgbn_from_cube else hypercube_counter, hypercube_counter))

			if train_with_patches and PATCH_SIZE != self.IMAGE_SIZE:
				print("Patch Size:\t\t\t\t{}\nNumber of Patches ({}/{} * {}):\t{}".format(self.PATCH_SIZE, self.IMAGE_SIZE, self.PATCH_SIZE, rgbn_counter, len(self.hypercubes)))

			print("Images Shape:\t\t\t\t{}\nHypercubes Shape:\t\t\t{}".format(list(self.images[0].size()), list(self.hypercubes[0].size()) if not lazy_read else "Lazy Read. Size calculated later."))

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
			image = image + 1.4007351398468018 + 0.1
			hypercube = hypercube + 2.335083484649658 if self.task == "reconstruction" else hypercube + 1.7359950542449951
			hypercube = hypercube + 0.1
			# image = (image-image.min())/(image.max() - image.min())
			# hypercube = (hypercube-hypercube.min())/(hypercube.max() - hypercube.min())

		return image, hypercube, classlabel, actual_classlabel

def get_dataloaders_reconstruction(trainset_size=0.7):
	dataset = DatasetFromDirectoryReconstruction(
		root=TEST_ROOT_DATASET_DIR,
		application_name=APPLICATION_NAME,
		patch_size=PATCH_SIZE,
		append_secondary_input=APPEND_SECONDARY_RGB_CAM_INPUT,
		augmentation_factor=0,
		verbose=True
	)
	train_data, valid_data = random_split(dataset, [int(trainset_size*len(dataset)), len(dataset) - int(len(dataset)*trainset_size)])

	print("Length of Training Set ({}%):\t\t{}".format(round(trainset_size * 100), len(train_data)))
	print("Length of Validation Set ({}%):\t\t{}".format(round((1-trainset_size) * 100), len(valid_data)))

	train_data_loader = DataLoader(dataset=train_data,
								   num_workers=4,
								   batch_size=batch_size,
								   shuffle=True,
								   pin_memory=True)

	valid_data_loader = DataLoader(dataset=valid_data,
								   num_workers=4,
								   batch_size=4,
								   shuffle=False,
								   pin_memory=True)

	return train_data_loader, valid_data_loader

class DatasetFromDirectoryReconstruction(Dataset):
	rgb_images, nir_images, secondary_rgb_images, hypercubes, patch_indices = [], [], [], [], {}
	def __init__(self, root, application_name=APPLICATION_NAME, patch_size=64, append_secondary_input=False, augmentation_factor=0, verbose=False):
		self.patch_size = patch_size
		self.augmentation_factor = augmentation_factor
		self.append_secondary_input = append_secondary_input

		image_width, image_height = 512, 512
		rgbn_counter, hypercube_counter, patch_index = 0, 0, 0

		print("Reading Images from:") if verbose else None

		for dataset in TEST_DATASETS:
			directory = os.path.join(root, application_name, "{}_204ch".format(dataset))
			print(" " * 19, "{0:62}".format(directory), "{} and {}".format(GT_RGBN_DIR_NAME, GT_SECONDARY_RGB_CAM_DIR_NAME) if append_secondary_input else GT_RGBN_DIR_NAME) if verbose else None
			for filename in glob(os.path.join(directory, "*.mat")):
				hypercube = load_mat(filename)
				hypercube = hypercube[:, :, BANDS]
				hypercube = (hypercube - hypercube.min()) / (hypercube.max() - hypercube.min())

				rgb_image = imread(os.path.join(directory, GT_RGBN_DIR_NAME, os.path.split(filename)[-1].replace(".mat", "_RGB.png")))
				nir_image = imread(os.path.join(directory, GT_RGBN_DIR_NAME, os.path.split(filename)[-1].replace(".mat", "_NIR.png")))
				secondary_rgb_image = imread(os.path.join(directory, GT_SECONDARY_RGB_CAM_DIR_NAME, os.path.split(filename)[-1].replace(".mat", "_RGB.png")))

				rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
				nir_image = (nir_image - nir_image.min()) / (nir_image.max() - nir_image.min())
				secondary_rgb_image = (secondary_rgb_image - secondary_rgb_image.min()) / (secondary_rgb_image.max() - secondary_rgb_image.min())

				self.rgb_images.append(rgb_image)
				self.nir_images.append(nir_image)
				self.secondary_rgb_images.append(secondary_rgb_image)
				self.hypercubes.append(hypercube)

				image_width, image_height = rgb_image.shape[0:2]

				for patch_i in range(0, image_width, self.patch_size):
					for patch_j in range(0, image_height, self.patch_size):
						self.patch_indices[patch_index] = (rgbn_counter, patch_i, patch_j)
						patch_index += 1

				rgbn_counter += 1
				hypercube_counter += 1

		self.dataset_size = len(self.rgb_images)

		self.patch_size = image_width if (self.patch_size <= 0 or self.patch_size > image_width) else self.patch_size
		self.dataset_size *= (image_width // self.patch_size)

		self.patch_size = image_height if (self.patch_size <= 0 or self.patch_size > image_height) else self.patch_size
		self.dataset_size *= (image_height // self.patch_size)

		if verbose:
			print("\nBands used:\t\t\t\t{}\nNumber of Bands:\t\t\t{}".format(BANDS, len(BANDS)))
			print("Number of RGBN Files:\t\t\t{}\nNumber of Hypercubes Files:\t\t{}".format(rgbn_counter, hypercube_counter))
			print("RGB Image Dataset Size:\t\t\t{}\nHypercube Dataset Size:\t\t\t{}".format(len(self.rgb_images), len(self.hypercubes)))
			print("Appended Secondary Input:\t\t{}".format("yes" if append_secondary_input else "no"))
			print("Images Shape:\t\t\t\t{}\nHypercubes Shape:\t\t\t{}".format(list(self.rgb_images[0].shape), list(self.hypercubes[0].shape)))
			if self.patch_size != image_height and self.patch_size != image_width and self.patch_size > 0:
				print("Patch Size:\t\t\t\t{}\nNumber of Patches ({}/{} * {}):\t{}".format(self.patch_size, min(image_width, image_height), self.patch_size, rgbn_counter, self.dataset_size))

		assert len(self.rgb_images) == len(self.nir_images) == len(self.secondary_rgb_images) == len(self.hypercubes), "Number of images and hypercubes do not match."

	def __len__(self):
		return self.dataset_size

	def __getitem__(self, index):
		data_idx, patch_i, patch_j = self.patch_indices[index]
		rgb_image = self.rgb_images[data_idx][patch_i:patch_i+self.patch_size, patch_j:patch_j+self.patch_size, :]
		nir_image = np.expand_dims(np.asarray(self.nir_images[data_idx]), 2)[patch_i:patch_i+self.patch_size, patch_j:patch_j+self.patch_size, :]
		secondary_rgb_image = self.secondary_rgb_images[data_idx][patch_i:patch_i+self.patch_size, patch_j:patch_j+self.patch_size, :]
		hypercube = self.hypercubes[data_idx][patch_i:patch_i+self.patch_size, patch_j:patch_j+self.patch_size, :]

		visualize_data_item(rgb_image, hypercube, secondary_rgb_image, 12, "0")

		rgb_image = np.transpose(rgb_image, [2, 0, 1])
		nir_image = np.transpose(nir_image, [2, 0, 1])
		secondary_rgb_image = np.transpose(secondary_rgb_image, [2, 0, 1])
		hypercube = np.transpose(hypercube, [2, 0, 1])

		use_secondary = bool(random.getrandbits(1)) if self.append_secondary_input else False

		image = np.concatenate((secondary_rgb_image, nir_image), axis=0) if use_secondary else np.concatenate((rgb_image, nir_image), axis=0)

		image = torch.from_numpy(image.copy()).float()
		hypercube = torch.from_numpy(hypercube.copy()).float()

		# print("Image Min: %f\tMax: %f\tHypercube Min: %f\tMax: %f" % (image.min(), image.max(), hypercube.min(), hypercube.max()))

		return image, hypercube + EPS

class DatasetFromDirectoryClassification(Dataset):
	hypercubes, labels = [], []
	def __init__(self, root, application_name=APPLICATION_NAME, augmentation_factor=0, mobile_reconstructed_folder=None, verbose=False):
		self.augmentation_factor = augmentation_factor

		image_width, image_height = 512, 512
		hypercube_counter = 0

		print("Reading Images from:") if verbose else None

		for dataset in TEST_DATASETS:
			directory = os.path.join(root, application_name, "{}_204ch".format(dataset))
			directory = os.path.join(directory, mobile_reconstructed_folder) if mobile_reconstructed_folder != None else directory
			print(" " * 19, "{0:62}".format(directory))
			for filename in glob(os.path.join(directory, "*.mat")):
				hypercube = load_mat(filename)
				hypercube = hypercube[:, :, BANDS]
				hypercube = (hypercube - hypercube.min()) / (hypercube.max() - hypercube.min())

				# TODO: Load only the patch of hypercube which is needed (from the Shelflife.csv file)

				self.hypercubes.append(hypercube)

				# TODO: Load the labels from the Shelflife.csv file

				image_width, image_height = hypercube.shape[0:2]

				hypercube_counter += 1

		self.dataset_size = len(self.hypercubes)

		if verbose:
			print("\nBands used:\t\t\t\t{}\nNumber of Bands:\t\t\t{}".format(BANDS, len(BANDS)))
			print("Number of Hypercubes Files:\t\t{}".format(hypercube_counter))
			print("Hypercube Dataset Size:\t\t\t{}".format(len(self.hypercubes)))
			print("Hypercubes Shape:\t\t\t\t{}\nLabels Shape:\t\t\t{}".format(list(self.hypercubes[0].shape), list(self.labels[0].shape)))

		assert len(self.hypercubes) == len(self.labels), "Number of hypercubes and labels do not match."

	def __len__(self):
		return self.dataset_size

	def __getitem__(self, index):
		hypercube = self.hypercubes[index]
		label = self.labels[index]

		# visualize_data_item(rgb_image, hypercube, None, 12, "0")

		hypercube = np.transpose(hypercube, [2, 0, 1])
		hypercube = torch.from_numpy(hypercube.copy()).float()

		label = torch.from_numpy(label.copy()).long()

		# print("Image Min: %f\tMax: %f\tHypercube Min: %f\tMax: %f" % (image.min(), image.max(), hypercube.min(), hypercube.max()))

		return hypercube + EPS, label