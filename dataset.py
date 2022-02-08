import os
import numpy as np

import h5py
from glob import glob
from imageio import imread
from utils import load_mat

import torch
from torch.utils.data import Dataset
from config import var_name

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