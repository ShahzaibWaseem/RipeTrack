import os
import sys
sys.path.append("..")

import imageio
from PIL import Image
from spectral import envi
from scipy.io import savemat

import numpy as np
import pandas as pd

import torch

from models.DeepWB import deepWBnet
from models.DeepWBUtils import deep_wb, colorTempInterpolate, to_image

from utils import create_directory
from config import GT_RGBN_DIR_NAME, GT_AUXILIARY_RGB_CAM_DIR_NAME, GT_HYPERCUBES_DIR_NAME, CAMERA_OUTPUT_ROOT_PATH,\
	TEST_ROOT_DATASET_DIR, APPLICATION_NAME, RGBN_BANDS, MODEL_PATH, DEEP_WB_DIR, SHELF_LIFE_GROUND_TRUTH_FILENAME,\
	device, var_name

import matplotlib.pyplot as plt

def normalize(band, min=-1, max=-1):
	if min ==-1 or max == -1:
		min = np.min(band)
		max = np.max(band)
	return (band - min) / (max - min)

def reverse_normalize(band, min, max, dtype=np.uint8):
	return (band * (max - min) + min).astype(dtype)

def reduce_noise(band):
	fourier_transform = np.fft.fft2(band)
	center_shift = np.fft.fftshift(fourier_transform)

	rows, cols = band.shape
	crow, ccol = rows // 2, cols // 2

	# horizontal mask
	center_shift[crow - 4:crow + 4, 0:ccol - 10] = 1
	center_shift[crow - 4:crow + 4, ccol + 10:] = 1

	f_shift = np.fft.ifftshift(center_shift)
	denoised_image = np.fft.ifft2(f_shift)
	denoised_image = np.real(denoised_image)

	return denoised_image

def denoise_image(image, dtype=np.uint8):
	image = np.expand_dims(image, axis=-1) if (len(image.shape) != 3) else image

	channels = image.shape[2]
	min, max = image.min(), image.max()
	normalized_image = normalize(image, min=min, max=max)

	removed_noise_image = np.zeros(image.shape)

	for band in range(channels):
		removed_noise_image[:, :, band] = reduce_noise(normalized_image[:, :, band])
	
	removed_noise_image = normalize(removed_noise_image)
	removed_noise_image = reverse_normalize(removed_noise_image, min, max, dtype)

	return removed_noise_image

class CommonLighting(torch.nn.Module):
	net_awb = deepWBnet()
	net_t = deepWBnet()
	net_s = deepWBnet()
	task = "all"
	S = 656

	def __init__(self):
		super().__init__()
		awb_model_path = os.path.join("..", MODEL_PATH, DEEP_WB_DIR, "net_awb.pth")
		t_model_path = os.path.join("..", MODEL_PATH, DEEP_WB_DIR, "net_t.pth")
		s_model_path = os.path.join("..", MODEL_PATH, DEEP_WB_DIR, "net_s.pth")

		if os.path.exists(awb_model_path) and os.path.exists(t_model_path) and os.path.exists(s_model_path):
			# load awb net
			self.net_awb = deepWBnet()
			print("Loading model {}".format(awb_model_path))
			self.net_awb.to(device=device)
			self.net_awb.load_state_dict(torch.load(awb_model_path, map_location=device))
			self.net_awb.eval()
			# load tungsten net
			self.net_t = deepWBnet()
			print("Loading model {}".format(t_model_path))
			self.net_t.to(device=device)
			self.net_t.load_state_dict(torch.load(t_model_path, map_location=device))
			self.net_t.eval()
			# load shade net
			self.net_s = deepWBnet()
			print("Loading model {}".format(s_model_path))
			self.net_s.to(device=device)
			self.net_s.load_state_dict(torch.load(s_model_path, map_location=device))
			self.net_s.eval()
			print("Models loaded !")

	def forward(self, image):
		image = Image.fromarray(image)
		_, out_t, out_s = deep_wb(image, task=self.task, net_awb=self.net_awb, net_s=self.net_s, net_t=self.net_t, device=device, s=self.S)
		_, out_d, _ = colorTempInterpolate(out_t, out_s)
		result_d = to_image(out_d)
		return result_d

def plotImages(rgb_image, nir_image, hypercube, rgb_secondary_image):
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(35, 30))
	ax1.imshow(rgb_image)
	ax1.title.set_text("RGB Image")
	ax2.imshow(nir_image)
	ax2.title.set_text("NIR Image")
	rgb_from_cube = hypercube[var_name][:,:, RGBN_BANDS[:3]]
	rgb_from_cube = hypercubeBandtoImage(rgb_from_cube)
	rgb_from_cube[:, :, [0, 2]] = rgb_from_cube[:, :, [2, 0]]
	ax3.imshow(rgb_secondary_image)
	ax3.title.set_text("RGB Secondary Image")
	ax4.imshow(rgb_from_cube)
	ax4.title.set_text("Hypercube")
	plt.show()

def hypercubeBandtoImage(image):
	"""
	Normalizes band chosen from hypercube to an image which can be saved
		Input:	Hyperspectral band	[0 - 4095]
		Output:	Image				[0 - 255]
	"""
	image = (image - image.min())/(image.max() - image.min())	# normalizing 	[0 - 4095]	-> [0.0 - 1.0]
	image *= 255												# scaling		[0.0 - 1.0]	-> [0 - 255]
	image = image.astype(np.uint8)
	return image

def main():
	root_directory = os.path.join("..", CAMERA_OUTPUT_ROOT_PATH)
	ground_truth_df = pd.read_csv(os.path.join(SHELF_LIFE_GROUND_TRUTH_FILENAME))
	commonLighting = CommonLighting()

	for index in ground_truth_df.index:
		hs_filenumbers = [int(x) for x in ground_truth_df["HS Files"][index].split(",")]
		dataset_name = ground_truth_df["Fruit"][index].lower() + "-" + ground_truth_df["Type"][index].lower()

		output_hypercube_directory = os.path.join("..", TEST_ROOT_DATASET_DIR, APPLICATION_NAME, "%s_204ch" % dataset_name)

		output_rgbn_directory = os.path.join(output_hypercube_directory, GT_RGBN_DIR_NAME)
		output_secondary_rgbn_directory = os.path.join(output_hypercube_directory, GT_AUXILIARY_RGB_CAM_DIR_NAME)
		create_directory(os.path.join(output_hypercube_directory, GT_HYPERCUBES_DIR_NAME))
		create_directory(output_rgbn_directory)
		create_directory(output_secondary_rgbn_directory)

		input_catalog_directory = os.path.join(root_directory, "%s" % APPLICATION_NAME)

		for hs_filenumber in hs_filenumbers:
			hs_filepath = os.path.join(input_catalog_directory, "%d" % hs_filenumber, "results", "REFLECTANCE_%s.hdr" % hs_filenumber)
			if (not os.path.exists(hs_filepath)):
				print("File, %s, in the .csv does not exist. Skipping" % hs_filepath)
				continue
			hypercube = envi.open(hs_filepath, hs_filepath.replace(".hdr", ".dat"))

			rgb_filepath = os.path.join(input_catalog_directory, "%d" % hs_filenumber, "results", "REFLECTANCE_%s.png" % hs_filenumber)
			rgb_image = imageio.imread(rgb_filepath)[:, :, :3]
			# rgb_image = denoise_image(rgb_image)

			rgb_secondary_filepath = os.path.join(input_catalog_directory, "%d" % hs_filenumber, "results", "RGBBACKGROUND_%d.png" % hs_filenumber)
			rgb_secondary_image = imageio.imread(rgb_secondary_filepath)[:, :, :3]
			rgb_secondary_image = Image.fromarray(rgb_secondary_image)
			rgb_secondary_image = np.asarray(rgb_secondary_image.resize((512, 512)))

			nir_image = np.expand_dims(np.rot90(hypercube.read_band(RGBN_BANDS[-1]), k=-1), axis=-1)
			# nir_image = denoise_image(nir_image, dtype=np.float32)
			nir_image = hypercubeBandtoImage(nir_image)

			hypercube = hypercube.load() * 256
			# hypercube = denoise_image(hypercube, dtype=np.float32)
			hypercube = {var_name: np.rot90(hypercube, k=-1)}

			# plotImages(rgb_image, nir_image, hypercube, rgb_secondary_image)

			print("[%30s] Hypercube Shape: %s [Range: %d - %3d], RGB Image Shape: %s [Range: %d - %3d], NIR Image Shape: %s [Range: %d - %3d], RGB Secondary: [Range: %d - %3d]"
	 			% (os.path.join(os.path.split(output_hypercube_directory)[-1], "%s.mat" % hs_filenumber), hypercube[var_name].shape, hypercube[var_name].min(), hypercube[var_name].max(),
				   rgb_image.shape, np.min(rgb_image), np.max(rgb_image), nir_image.shape, np.min(nir_image), np.max(nir_image), np.min(rgb_secondary_image), np.max(rgb_secondary_image)))
			savemat(os.path.join(output_hypercube_directory, GT_HYPERCUBES_DIR_NAME, "%s.mat" % hs_filenumber), hypercube)
			
			daylight = commonLighting(rgb_image)
			imageio.imwrite(os.path.join(output_rgbn_directory, "%s_RGB.png" % hs_filenumber), rgb_image)
			imageio.imwrite(os.path.join(output_rgbn_directory, "%s_RGB-D.png" % hs_filenumber), daylight)
			imageio.imwrite(os.path.join(output_rgbn_directory, "%s_NIR.png" % hs_filenumber), nir_image)

			daylight = commonLighting(rgb_secondary_image)
			imageio.imwrite(os.path.join(output_secondary_rgbn_directory, "%s_RGB.png" % hs_filenumber), rgb_secondary_image)
			imageio.imwrite(os.path.join(output_secondary_rgbn_directory, "%s_RGB-D.png" % hs_filenumber), daylight)
			imageio.imwrite(os.path.join(output_secondary_rgbn_directory, "%s_NIR.png" % hs_filenumber), nir_image)

if __name__ == "__main__":
	create_directory(os.path.join("..", MODEL_PATH, DEEP_WB_DIR))
	main()