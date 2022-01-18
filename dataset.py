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
		self.data = hf.get("data")
		self.target = hf.get("label")

	def __getitem__(self, index):
		return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()
		
	def __len__(self):
		return self.data.shape[0]

class DatasetFromDirectory(Dataset):
	def __init__(self, root):
		self.root = root
		self.var_name = var_name
		self.labels = []
		self.images = []
		for filename in glob(os.path.join(self.root, "RGB", "*.png")):
			mat_file_name = filename.split("/")[-1].split("_")[0]
			rgb_img_path = filename
			nir_img_path = os.path.join(self.root, "NIR", filename.split("/")[-1].replace("RGB", "NIRc"))
			
			rgb = imread(rgb_img_path)
			rgb = rgb/255

			nir = imread(nir_img_path)
			nir = nir/255

			image = np.dstack((rgb, nir))

			ground_t = load_mat(os.path.join(self.root, "mat", mat_file_name + ".mat"), self.var_name)
			ground_t = ground_t[self.var_name][:,:,1:204:4] / 4095

			self.images.append(image)
			self.labels.append(ground_t)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		return self.images[index], self.labels[index]