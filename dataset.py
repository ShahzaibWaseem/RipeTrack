import os
import time
import random
from collections import OrderedDict

import numpy as np
import pandas as pd

import h5py
from glob import glob
from imageio import imread

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split

from utils import load_mat, load_mat_patched, read_image, data_augmentation, get_normalization_parameters, visualize_data_item
from config import GT_RGBN_DIR_NAME, GT_REMOVED_IR_CUTOFF_DIR_NAME, GT_AUXILIARY_RGB_CAM_DIR_NAME, GT_HYPERCUBES_DIR_NAME, RECONSTRUCTED_HS_DIR_NAME,\
	MOBILE_DATASET_DIR_NAME, MOBILE_RECONSTRUCTED_HS_DIR_NAME, OP_MOBILE_DATASET_DIR_NAME, PATCHED_HS_DIR_NAME, TRAIN_VAL_TEST_SPLIT_DIR_NAME,\
	PATCHED_INFERENCE, BANDS, BANDS_WAVELENGTHS, BAND_SPACING, RGBN_BANDS, NIR_BANDS, TEST_DATASETS, TRAIN_DATASET_DIR, TEST_ROOT_DATASET_DIR, DATA_PREP_PATH,\
	TRAIN_DATASET_FILES, VALID_DATASET_FILES, APPEND_SECONDARY_RGB_CAM_INPUT, APPLICATION_NAME, IMAGE_SIZE, PATCH_SIZE, CLASSIFICATION_PATCH_SIZE, STRIDE,\
	SHELF_LIFE_GROUND_TRUTH_FILENAME, GT_DATASET_CROPS_FILENAME, MOBILE_DATASET_CROPS_FILENAME, LABELS_DICT, SUB_LABELS_DICT, TIME_LEFT_DICT, FRUITS_DICT, EPS,\
	batch_size, device, use_mobile_dataset

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

def get_dataloaders_reconstruction():
	movePixels = 10
	# reconstructionTransforms = transforms.Compose([Misalign(movePixels=movePixels, imageSize=IMAGE_SIZE, newImageSize=IMAGE_SIZE-(2*movePixels))])
	reconstructionTransforms = None
	train_data = DatasetFromDirectoryReconstructionTrain(
		root=TEST_ROOT_DATASET_DIR,
		application_name=APPLICATION_NAME,
		patch_size=PATCH_SIZE,
		augment=False,
		movePixels=0 if reconstructionTransforms == None else movePixels,
		stride=PATCH_SIZE,
		use_auxiliary_input=APPEND_SECONDARY_RGB_CAM_INPUT,
		transforms=reconstructionTransforms,
		verbose=True
	)

	valid_data = DatasetFromDirectoryReconstructionValid(
		root=TEST_ROOT_DATASET_DIR,
		application_name=APPLICATION_NAME,
		use_auxiliary_input=APPEND_SECONDARY_RGB_CAM_INPUT,
		verbose=False
	)

	train_data_loader = DataLoader(dataset=train_data,
								   num_workers=2,
								   batch_size=batch_size,
								   shuffle=True,
								   pin_memory=True,
								   drop_last=True)

	valid_data_loader = DataLoader(dataset=valid_data,
								   num_workers=2,
								   batch_size=1,
								   shuffle=False,
								   pin_memory=True)

	return train_data_loader, valid_data_loader

class DatasetFromDirectoryReconstructionTrain(Dataset):
	rgbn_images, hypercubes = [], []

	def __init__(self, root, application_name=APPLICATION_NAME, patch_size=64, augment=True, movePixels=10, stride=8, use_auxiliary_input=False, transforms=None, verbose=False):
		self.patch_size = patch_size
		self.augment = augment
		self.use_auxiliary_input = use_auxiliary_input
		image_width, image_height = IMAGE_SIZE, IMAGE_SIZE
		rgbn_counter, hypercube_counter = 0, 0
		self.movePixels = movePixels
		self.transforms = transforms
		self.stride = stride
		self.patch_per_line = (image_width-patch_size)//stride+1
		self.patch_per_column = (image_height-patch_size)//stride+1
		self.patch_per_image = self.patch_per_line*self.patch_per_column

		print("Reading Images from:") if verbose else None

		for dataset in TEST_DATASETS:
			with open(os.path.join(root, application_name, "%s_204ch" % dataset, TRAIN_VAL_TEST_SPLIT_DIR_NAME, "train.txt"), "r") as train_file:
				hypercube_list = [str(filename).replace("\n", ".mat") for filename in train_file]
			directory = os.path.join(root, application_name, "{}_204ch".format(dataset))
			print(" " * 19, "{0:62}".format(directory), "{} and {}".format(GT_RGBN_DIR_NAME, GT_AUXILIARY_RGB_CAM_DIR_NAME) if use_auxiliary_input else GT_RGBN_DIR_NAME, end="\t") if verbose else None
			dataset_load_time = time.time()
			for filename in hypercube_list:
				if int(os.path.split(filename)[-1].split(".")[0]) % 2 != 0:
					continue
				image_width, image_height = IMAGE_SIZE, IMAGE_SIZE
				hypercube = load_mat(os.path.join(directory, GT_HYPERCUBES_DIR_NAME, filename))
				# nir_image = np.float32(hypercube[:, :, random.choices(NIR_BANDS)])
				hypercube = hypercube[:, :, BANDS]
				# hypercube = hypercube[movePixels:image_width-movePixels, movePixels:image_height-movePixels, :] if not self.transforms == None else hypercube
				hypercube = (hypercube - hypercube.min()) / (hypercube.max() - hypercube.min())
				hypercube = np.transpose(hypercube, [2, 0, 1])
				hypercube += EPS
				hypercube_counter += 1
				self.hypercubes.append(hypercube)

				rgb_image = np.float32(imread(os.path.join(directory, GT_RGBN_DIR_NAME if not use_auxiliary_input else GT_AUXILIARY_RGB_CAM_DIR_NAME, os.path.split(filename)[-1].replace(".mat", "_RGB%s.png" % "-D"))))
				rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
				# rgb_image = self.transforms(rgb_image) if not self.transforms == None else rgb_image

				nir_image = np.float32(imread(os.path.join(directory, GT_RGBN_DIR_NAME, os.path.split(filename)[-1].replace(".mat", "_NIR.png"))))
				nir_image = (nir_image - nir_image.min()) / (nir_image.max() - nir_image.min())
				# nir_image = nir_image[movePixels:image_width-movePixels, movePixels:image_height-movePixels] if not self.transforms == None else nir_image
				nir_image = np.expand_dims(np.asarray(nir_image), 2)
				nir_image = np.transpose(nir_image, [2, 0, 1])

				# image = rgb_image
				image = np.dstack((rgb_image, nir_image))
				image = np.transpose(image, [2, 0, 1])
				rgbn_counter += 1

				self.rgbn_images.append(image)

				image_width, image_height = image.shape[1], image.shape[2]
			print("{:>4} s".format(round(time.time()-dataset_load_time))) if verbose else None

		self.dataset_size = len(self.rgbn_images) * self.patch_per_image

		if verbose:
			print("Bands used:".ljust(40), BANDS)
			print("Actual Bands:".ljust(40), BANDS_WAVELENGTHS)
			print("Number of Bands:".ljust(40), len(BANDS))
			print("Number of RGBN Files:".ljust(40), rgbn_counter)
			print("Number of Hypercubes Files:".ljust(40), hypercube_counter)
			print("RGB Image Dataset Size:".ljust(40), len(self.rgbn_images))
			print("Hypercube Dataset Size:".ljust(40), len(self.hypercubes))
			print("Used Auxiliary Input:".ljust(40), ("yes" if use_auxiliary_input else "no"))
			print("RGBN Image Shape:".ljust(40), list(self.rgbn_images[0].shape))
			print("Hypercubes Shape:".ljust(40), list(self.hypercubes[0].shape))
			if self.patch_size != image_height and self.patch_size != image_width and self.patch_size > 0:
				print("Patch Size:".ljust(40), self.patch_size)
				print("Number of Patches ({:<2}/{:<3} * {:<2}):\t".ljust(40).format(self.patch_size, min(image_width, image_height), self.patch_size, rgbn_counter), self.dataset_size)

		assert len(self.rgbn_images) == len(self.hypercubes), "Number of images and hypercubes do not match."

	def __len__(self):
		return self.dataset_size

	def augmentation(self, image, rotTimes, vFlip, hFlip):
		# Random rotation
		for _ in range(rotTimes):
			image = np.rot90(image.copy(), axes=(1, 2))
		# Random vertical Flip
		for _ in range(vFlip):
			image = image[:, :, ::-1].copy()
		# Random horizontal Flip
		for _ in range(hFlip):
			image = image[:, ::-1, :].copy()
		return image

	def __getitem__(self, index):
		stride = self.stride
		patch_size = self.patch_size
		image_idx, patch_idx = index // self.patch_per_image, index % self.patch_per_image
		h_idx, w_idx = patch_idx // self.patch_per_line, patch_idx % self.patch_per_line
		rgbn_image = self.rgbn_images[image_idx]
		hypercube = self.hypercubes[image_idx]

		rgbn_image = rgbn_image[:, h_idx*stride:h_idx*stride+patch_size, w_idx*stride:w_idx*stride+patch_size]
		hypercube = hypercube[:, h_idx*stride:h_idx*stride+patch_size, w_idx*stride:w_idx*stride+patch_size]

		if self.augment:
			rotTimes = random.randint(0, 3)
			vFlip = random.randint(0, 1)
			hFlip = random.randint(0, 1)
			rgbn_image = self.augmentation(rgbn_image, rotTimes, vFlip, hFlip)
			hypercube = self.augmentation(hypercube, rotTimes, vFlip, hFlip)

		# visualize_data_item(np.transpose(rgb_image.numpy(), [1, 2, 0]), np.transpose(hypercube.numpy(), [1, 2, 0]), np.transpose(secondary_rgb_image.numpy(), [1, 2, 0]), 12, 0)
		# print(rgb_image.shape, nir_image.shape, secondary_rgb_image.shape if self.use_auxiliary_input else None, hypercube.shape)
		# print("Image Min: %f\tMax: %f\tHypercube Min: %f\tMax: %f" % (image.min(), image.max(), hypercube.min(), hypercube.max()))

		return np.ascontiguousarray(rgbn_image), np.ascontiguousarray(hypercube)

class DatasetFromDirectoryReconstructionValid(Dataset):
	rgbn_images, hypercubes = [], []

	def __init__(self, root, application_name=APPLICATION_NAME, use_auxiliary_input=False, verbose=False):
		for dataset in TEST_DATASETS:
			with open(os.path.join(root, application_name, "%s_204ch" % dataset, TRAIN_VAL_TEST_SPLIT_DIR_NAME, "validation.txt"), "r") as valid_file:
				hypercube_list = [filename.replace("\n", ".mat") for filename in valid_file]
			directory = os.path.join(root, application_name, "{}_204ch".format(dataset))
			print(" " * 19, "{0:62}".format(directory), "{} and {}".format(GT_RGBN_DIR_NAME, GT_AUXILIARY_RGB_CAM_DIR_NAME) if use_auxiliary_input else GT_RGBN_DIR_NAME, end="\t") if verbose else None
			dataset_load_time = time.time()
			for filename in hypercube_list:
				hypercube = load_mat(os.path.join(directory, GT_HYPERCUBES_DIR_NAME, filename))
				# nir_image = np.float32(hypercube[:, :, random.choice(NIR_BANDS)])
				hypercube = hypercube[:, :, BANDS]
				hypercube = (hypercube - hypercube.min()) / (hypercube.max() - hypercube.min())
				hypercube = np.transpose(hypercube, [2, 0, 1])
				hypercube += EPS
				self.hypercubes.append(hypercube)

				rgb_image = np.float32(imread(os.path.join(directory, GT_RGBN_DIR_NAME if not use_auxiliary_input else GT_AUXILIARY_RGB_CAM_DIR_NAME, os.path.split(filename)[-1].replace(".mat", "_RGB%s.png" % "-D"))))
				rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

				nir_image = np.float32(imread(os.path.join(directory, GT_RGBN_DIR_NAME, os.path.split(filename)[-1].replace(".mat", "_NIR.png"))))
				nir_image = (nir_image - nir_image.min()) / (nir_image.max() - nir_image.min())
				nir_image = np.expand_dims(np.asarray(nir_image), 2)
				nir_image = np.transpose(nir_image, [2, 0, 1])

				# image = rgb_image
				image = np.dstack((rgb_image, nir_image))
				image = np.transpose(image, [2, 0, 1])

				self.rgbn_images.append(image)

			print("{:>4} s".format(round(time.time()-dataset_load_time))) if verbose else None

	def __getitem__(self, idx):
		hypercube = self.hypercubes[idx]
		rgbn_image = self.rgbn_images[idx]
		return np.ascontiguousarray(rgbn_image), np.ascontiguousarray(hypercube)

	def __len__(self):
		return len(self.hypercubes)

class Misalign(object):
	def __init__(self, movePixels=32, imageSize=512, newImageSize=448):
		self.movePixels = movePixels
		self.imageSize = imageSize
		self.newImageSize = newImageSize
		self.startingPixel = (self.imageSize - self.newImageSize) // 2

	def __call__(self, image):
		misaligned = image.copy()
		moveXpixels = random.randint(-self.movePixels, self.movePixels)
		moveYpixels = random.randint(-self.movePixels, self.movePixels)
		misaligned = misaligned[self.startingPixel+moveXpixels:self.startingPixel+moveXpixels+self.newImageSize, self.startingPixel+moveYpixels:self.startingPixel+moveYpixels+self.newImageSize, :]
		return misaligned

class_sizes = OrderedDict([(label, 0) for label in LABELS_DICT.keys()])
subclass_sizes = OrderedDict([(label, 0) for label in TIME_LEFT_DICT.values()])

def get_dataloaders_classification(trainset_size=0.7):
	global class_sizes

	classificationTransforms = transforms.Compose([FlipHorizontal(), FlipVertical(), Rotate()])

	dataset = DatasetFromDirectoryClassification(
		root=TEST_ROOT_DATASET_DIR,
		application_name=APPLICATION_NAME,
		hypercube_directory=MOBILE_RECONSTRUCTED_HS_DIR_NAME if use_mobile_dataset else None,
		transforms=classificationTransforms,
		patched_inference=PATCHED_INFERENCE,
		verbose=True
	)
	train_indices, valid_indices = dataset.divide_train_test(trainset_size)

	dataset_size = len(dataset)
	train_data, valid_data = random_split(dataset, [int(trainset_size*dataset_size), dataset_size - int(dataset_size*trainset_size)])
	print("Length of Training Set ({:<2}%):\t".ljust(40).format(round(trainset_size * 100)), len(train_data))
	print("Length of Validation Set ({:<2}%):\t".ljust(40).format(round((1-trainset_size) * 100)), len(valid_data))

	# class_weights = [1 / i for i in list(class_sizes.values())]
	# print("Class Weights Random Sampler:".ljust(40), class_weights)
	# print("Class Counts:".ljust(40), class_sizes)
	# sampler = WeightedRandomSampler(weights=class_weights, num_samples=dataset.__len__(), replacement=True)

	train_data_loader = DataLoader(dataset=train_data,
								   num_workers=4,
								   batch_size=batch_size,
								   shuffle=True,
								   pin_memory=True)
	if trainset_size < 1.0:
		valid_data_loader = DataLoader(dataset=valid_data,
								 	   num_workers=4,
									   batch_size=16,
									   shuffle=True,
									   pin_memory=True)
	else:
		valid_data_loader = None

	return train_data_loader, valid_data_loader

class DatasetFromDirectoryClassification(Dataset):
	images, hypercubes, labels, sublabels, fruits = [], [], [], [], []

	def __init__(self, root, application_name=APPLICATION_NAME, hypercube_directory=None, patched_inference=True, transforms=None, verbose=False):
		global class_sizes, subclass_sizes
		self.transforms = transforms

		image_width, image_height = 512, 512
		self.patch_size = CLASSIFICATION_PATCH_SIZE
		self.stride = STRIDE
		hypercube_counter = 0

		crops_df = pd.read_csv(os.path.join(DATA_PREP_PATH, MOBILE_DATASET_CROPS_FILENAME if hypercube_directory == MOBILE_RECONSTRUCTED_HS_DIR_NAME else GT_DATASET_CROPS_FILENAME))
		shelflife_df = pd.read_csv(os.path.join(DATA_PREP_PATH, SHELF_LIFE_GROUND_TRUTH_FILENAME))
		crops_df["w"] = crops_df["xmax"] - crops_df["xmin"]
		crops_df["h"] = crops_df["ymax"] - crops_df["ymin"]
		min_w, min_h = int(crops_df["w"].min()), int(crops_df["h"].min())			# Min Width:  89, Min Height:  90
		max_w, max_h = int(crops_df["w"].max()), int(crops_df["h"].max())			# Max Width: 214, Max Height: 220

		print("Reading Images from:") if verbose else None
		for dataset in TEST_DATASETS:
			directory = os.path.join(root, application_name, "{}_204ch".format(dataset))
			directory = os.path.join(directory, hypercube_directory) if hypercube_directory != None else os.path.join(directory, GT_HYPERCUBES_DIR_NAME)
			print("{0:21}".format(os.path.split(directory)[-1] if hypercube_directory == None else dataset), end=":")
			fruit_name_capt = shelflife_df["Fruit"].str.contains(dataset.split("-")[0].capitalize())
			friut_type_capt = shelflife_df["Type"].str.contains(dataset.split("-")[1].capitalize())
			# sub_labels_print_dict = shelflife_df[fruit_name_capt & friut_type_capt]["Remaining Life"].value_counts()b[shelflife_df["Remaining Life"].unique()].to_dict()
			print(shelflife_df[fruit_name_capt & friut_type_capt]["Shelf Life Label"].value_counts()[shelflife_df["Shelf Life Label"].unique()].to_dict(), end=", ")
			# print({list(TIME_LEFT_DICT.keys()).index(key): value for key, value in sub_labels_print_dict.items()}, end=", ")
			# print(sub_labels_print_dict, end=", ")
			dataset_load_time = time.time()
			for filename in glob(os.path.join(directory, "*.mat")):
				hypercube_number = os.path.split(filename)[-1].split(".")[0].split("_")[0]
				shelflife_record = shelflife_df[shelflife_df["HS Files"].str.contains(hypercube_number)].iloc[0]
				if shelflife_record["Skip"] == "yes": continue

				# rgb_image = imread(os.path.join(directory, MOBILE_DATASET_DIR_NAME, os.path.split(filename)[-1].replace(".mat", "_RGB.png")))
				# rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
				# rgb_image = np.transpose(rgb_image, [2, 0, 1])
				# image = rgb_image
				# rgb_image = np.expand_dims(rgb_image, axis=0)

				# nir_image = imread(os.path.join(directory, MOBILE_DATASET_DIR_NAME, os.path.split(filename)[-1].replace(".mat", "_NIR.png")))
				# nir_image = (nir_image - nir_image.min()) / (nir_image.max() - nir_image.min())
				# nir_image = np.expand_dims(np.asarray(nir_image), 2)
				# nir_image = np.transpose(nir_image, [2, 0, 1])
				# nir_image = np.expand_dims(nir_image, axis=0)
				# image = np.concatenate((rgb_image, nir_image), axis=0)
				# print(rgb_image.shape, nir_image.shape, image.shape)

				hypercube = load_mat(filename)
				hypercube = hypercube[:, :, BANDS] if hypercube_directory != None else hypercube
				hypercube = (hypercube - hypercube.min()) / (hypercube.max() - hypercube.min())
				hypercube = np.transpose(hypercube, [2, 0, 1]) + EPS

				crop_record = crops_df[crops_df["image"].isin(["{}_RGB.png".format(hypercube_number)])]
				xmin = int(crop_record["xmin"].iloc[0])
				ymin = int(crop_record["ymin"].iloc[0])
				xmax = int(crop_record["xmax"].iloc[0])
				ymax = int(crop_record["ymax"].iloc[0])

				label_name = shelflife_record["Shelf Life Label"]
				sublabel_name = shelflife_record["Remaining Life"]
				label = LABELS_DICT.get(label_name)
				sublabel = TIME_LEFT_DICT.get(sublabel_name)
				fruit_name = "{} {}".format(shelflife_record["Fruit"], shelflife_record["Type"])

				for patch_i in range(xmin, xmax, self.stride):
					if patch_i+self.patch_size > xmax: continue
					for patch_j in range(ymin, ymax, self.stride):
						if patch_j+self.patch_size > ymax: continue
						# imageCrop = image[:, patch_i:patch_i+self.patch_size, patch_j:patch_j+self.patch_size]
						# if (imageCrop.shape[1:3] != (self.patch_size, self.patch_size)):
						# 	continue
						# self.images.append(imageCrop)

						hypercubeCrop = hypercube[:, patch_i:patch_i+self.patch_size, patch_j:patch_j+self.patch_size]
						if (hypercubeCrop.shape[1:3] != (self.patch_size, self.patch_size)):
							continue
						self.hypercubes.append(hypercubeCrop)
						self.labels.append(label)
						self.sublabels.append(sublabel)
						self.fruits.append(fruit_name)
						class_sizes[label_name] += 1
						subclass_sizes[sublabel] += 1

				if patched_inference:
					patched_filename = os.path.join(directory, PATCHED_HS_DIR_NAME, os.path.split(filename)[-1])
					hypercube_combined = load_mat_patched(patched_filename)
					for hypercube in hypercube_combined.values():
						hypercube = np.transpose(hypercube, [2, 0, 1]) + EPS
						self.hypercubes.append(hypercube)
						self.labels.append(label)
						self.sublabels.append(sublabel)
						self.fruits.append(fruit_name)
						class_sizes[label_name] += 1
						subclass_sizes[sublabel] += 1

				hypercube_counter += 1
			print("{:>3}s".format(round(time.time()-dataset_load_time)))

		self.dataset_size = len(self.hypercubes)

		if verbose:
			print("Bands used:".ljust(40), BANDS)
			print("Actual Bands:".ljust(40), BANDS_WAVELENGTHS)
			print("Number of Bands:".ljust(40), len(BANDS))
			print("Number of Hypercubes Files:".ljust(40), hypercube_counter)
			# print("RGB+NIR Dataset Size:".ljust(40), len(self.images))
			print("Using Patched Hypercubes:".ljust(40), patched_inference)
			print("Hypercube Dataset Size:".ljust(40), len(self.hypercubes))
			print("Labels Dataset Size:".ljust(40), len(self.labels))
			print("Sublabels Dataset Size:".ljust(40), len(self.sublabels))
			# print("Images Shape:".ljust(40), list(self.images[0].shape))
			print("Hypercubes Shape:".ljust(40), list(self.hypercubes[0].shape))
			print("Width Range:".ljust(40), "{} - {}".format(min_w, max_w))
			print("Height Range:".ljust(40), "{} - {}".format(min_h, max_h))
			print("Class Label Sizes:".ljust(40), class_sizes)
			print("Subclass Label Sizes:".ljust(40), subclass_sizes)

		assert len(self.hypercubes) == len(self.labels), "Number of hypercubes and labels do not match."

	def divide_train_test(self, trainset_size=0.7):
		# indices = list(range(len(self.images)))
		indices = list(range(len(self.hypercubes)))
		trainset_size_indices = int(trainset_size*len(indices))
		# train_data, valid_data = random_split(indices, [int(trainset_size*len(indices)), len(indices) - int(len(indices)*trainset_size)])
		train_data, valid_data = Subset(indices, range(trainset_size_indices)), Subset(indices, range(trainset_size_indices, len(indices)))		# Non Random Split
		self.train_indices, self.valid_indices = train_data.indices, valid_data.indices
		return train_data, valid_data

	def __len__(self):
		return self.dataset_size

	def getLabels(self):
		return self.labels

	def __getitem__(self, index):
		# image = self.images[index]
		# if self.transforms is not None and index in self.train_indices:
		# 	image = self.transforms(image)

		hypercube = self.hypercubes[index]
		if self.transforms is not None and index in self.train_indices:
			hypercube = self.transforms(hypercube)
		label = torch.tensor(self.labels[index])
		sublabel = torch.tensor(self.sublabels[index])
		fruit = self.fruits[index]

		# visualize_data_item(None, hypercube, None, 12, "0")
		# image = torch.tensor(image.copy()).float()
		hypercube = torch.tensor(hypercube.copy()).float()
		# print("Image Min: %f\tMax: %f\tHypercube Min: %f\tMax: %f" % (image.min(), image.max(), hypercube.min(), hypercube.max()))

		return hypercube, label, sublabel, fruit

class FlipHorizontal(object):
	def __init__(self, p=0.5):
		self.p = p

	def __call__(self, image):
		if random.random() < self.p:
			return image[:, ::-1, :].copy()
		return image

class FlipVertical(object):
	def __init__(self, p=0.5):
		self.p = p

	def __call__(self, image):
		if random.random() < self.p:
			return image[::-1, :, :].copy()
		return image

class Rotate(object):
	def __init__(self, p=0.5):
		self.p = p

	def __call__(self, image):
		rotated = image.copy()
		if random.random() < self.p:
			for _ in range(random.randint(0, 3)):
				rotated = np.rot90(rotated, axes=(1, 2))
		return rotated