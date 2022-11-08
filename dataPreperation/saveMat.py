import os
import sys
sys.path.append("..")

import numpy as np
from glob import glob

import imageio
from scipy.io import savemat
from spectral import *

from config import CAMERA_OUTPUT_ROOT_PATH, EXTRACT_DATASETS, TEST_ROOT_DATASET_DIR, DATASET_NAME, RGBN_BANDS, NORMALIZATION_FACTOR, var_name, create_directory

import matplotlib.pyplot as plt

if __name__ == "__main__":
	root_directory = os.path.join("..", CAMERA_OUTPUT_ROOT_PATH)

	for dataset_name in EXTRACT_DATASETS:
		output_hypercube_directory = os.path.join("..", TEST_ROOT_DATASET_DIR, "working_%s" % DATASET_NAME, "working_%s_204ch" % dataset_name)
		output_rgbn_directory = os.path.join("..", TEST_ROOT_DATASET_DIR, "RGBNIRImages", "working_%s" % DATASET_NAME, "working_%s_204ch" % dataset_name)
		create_directory(output_hypercube_directory)
		create_directory(output_rgbn_directory)

		for categories in sorted(glob(os.path.join(root_directory, "working_%s" % dataset_name, "*"))):
			hypercube_filename = categories.split("/")[-1]
			hypercube_filepath = os.path.join(categories, "results", "REFLECTANCE_%s.hdr" % hypercube_filename)
			rgb_filepath = os.path.join(categories, "%s.png" % hypercube_filename)
			rgb_image = imageio.imread(rgb_filepath)[:, :, :3]

			# hypercube_image = open_image(hypercube_filepath)
			hypercube_image = envi.open(hypercube_filepath, hypercube_filepath.replace(".hdr", ".dat"))
			nir_image = np.expand_dims(np.rot90(hypercube_image.read_band(RGBN_BANDS[-1]), k=-1), axis=-1)*256
			hypercube_image = hypercube_image.load()*256
			hypercube_image = {var_name: np.rot90(hypercube_image, k=-1)}

			# print(nir_image.dtype)
			# nir_image = ((nir_image - nir_image.min()) * (1/(nir_image.max() - nir_image.min()) * 255)).astype(np.uint8)
			# nir_image = (nir_image * (1/nir_image.max() * 255)).astype(np.uint8)
			print("RGB: %d, %d\tNIR: %f, %f\tHS: %f, %f" % (np.min(rgb_image), np.max(rgb_image), np.min(nir_image), np.max(nir_image), hypercube_image[var_name].min(), hypercube_image[var_name].max()))

			# print(nir_image)
			# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
			# ax1.imshow(rgb_image)
			# ax2.imshow(nir_image)
			# rgb_from_cube = hypercube_image[var_name][:,:, RGBN_BANDS[:3]]/256
			# rgb_from_cube[:, :, [0, 2]] = rgb_from_cube[:, :, [2, 0]]
			# ax3.imshow(rgb_from_cube)
			# plt.show()

			print("[%42s] Hypercube Shape: %s, RGB Image Shape: %s, NIR Image Shape: %s" % (os.path.join(categories.split("/")[-2], "%s.mat" % hypercube_filename), hypercube_image[var_name].shape, rgb_image.shape, nir_image.shape))
			savemat(os.path.join(output_hypercube_directory, "%s.mat" % hypercube_filename), hypercube_image)

			imageio.imwrite(os.path.join(output_rgbn_directory, "%s_RGB.png" % hypercube_filename), rgb_image)
			imageio.imwrite(os.path.join(output_rgbn_directory, "%s_NIR.png" % hypercube_filename), nir_image)