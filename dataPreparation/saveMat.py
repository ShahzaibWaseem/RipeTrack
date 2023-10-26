import os
import sys
sys.path.append("..")

import imageio
from PIL import Image

import numpy as np
import pandas as pd

from spectral import envi
from scipy.io import savemat

from config import CAMERA_OUTPUT_ROOT_PATH, TEST_ROOT_DATASET_DIR, APPLICATION_NAME, RGBN_BANDS, SHELF_LIFE_GROUND_TRUTH_FILENAME, GT_RGBN_DIR_NAME, GT_SECONDARY_RGB_CAM_DIR_NAME, var_name, create_directory

import matplotlib.pyplot as plt

def plotImages(rgb_image, nir_image, hypercube, rgb_secondary_image):
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(35, 30))
	ax1.imshow(rgb_image)
	ax1.title.set_text("RGB Image")
	ax2.imshow(nir_image)
	ax2.title.set_text("NIR Image")
	rgb_from_cube = hypercube[var_name][:,:, RGBN_BANDS[:3]]
	rgb_from_cube = bandToNpNormalize(rgb_from_cube)
	rgb_from_cube[:, :, [0, 2]] = rgb_from_cube[:, :, [2, 0]]
	ax3.imshow(rgb_secondary_image)
	ax3.title.set_text("RGB Secondary Image")
	ax4.imshow(rgb_from_cube)
	ax4.title.set_text("Hypercube")
	plt.show()

def bandToNpNormalize(image):
	image = (image - image.min())/(image.max() - image.min())
	image *= 255
	image = image.astype(np.uint8)
	return image

def main():
	root_directory = os.path.join("..", CAMERA_OUTPUT_ROOT_PATH)
	ground_truth_df = pd.read_csv(os.path.join(SHELF_LIFE_GROUND_TRUTH_FILENAME))

	for index in ground_truth_df.index:
		hs_filenumbers = [int(x) for x in ground_truth_df["HS Files"][index].split(",")]
		dataset_name = ground_truth_df["Fruit"][index].lower() + "-" + ground_truth_df["Type"][index].replace("'", "").lower()

		output_hypercube_directory = os.path.join("..", TEST_ROOT_DATASET_DIR, APPLICATION_NAME, "%s_204ch" % dataset_name)
		output_rgbn_directory = os.path.join(output_hypercube_directory, GT_RGBN_DIR_NAME)
		output_secondary_rgbn_directory = os.path.join(output_hypercube_directory, GT_SECONDARY_RGB_CAM_DIR_NAME)
		create_directory(output_hypercube_directory)
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

			rgb_secondary_filepath = os.path.join(input_catalog_directory, "%d" % hs_filenumber, "results", "RGBBACKGROUND_%s.png" % hs_filenumber)
			rgb_secondary_image = imageio.imread(rgb_secondary_filepath)[:, :, :3]
			rgb_secondary_image = Image.fromarray(rgb_secondary_image)
			rgb_secondary_image = np.asarray(rgb_secondary_image.resize((512, 512)))

			nir_image = np.expand_dims(np.rot90(hypercube.read_band(RGBN_BANDS[-1]), k=-1), axis=-1)
			nir_image = bandToNpNormalize(nir_image)

			hypercube = hypercube.load()*256
			hypercube = {var_name: np.rot90(hypercube, k=-1)}

			# plotImages(rgb_image, nir_image, hypercube, rgb_secondary_image)

			print("[%30s] Hypercube Shape: %s [Range: %d - %3d], RGB Image Shape: %s [Range: %d - %3d], NIR Image Shape: %s [Range: %d - %3d], RGB Secondary: [Range: %d - %3d]"
	 			% (os.path.join(os.path.split(output_hypercube_directory)[-1], "%s.mat" % hs_filenumber), hypercube[var_name].shape, hypercube[var_name].min(), hypercube[var_name].max(),
				   rgb_image.shape, np.min(rgb_image), np.max(rgb_image), nir_image.shape, np.min(nir_image), np.max(nir_image), np.min(rgb_secondary_image), np.max(rgb_secondary_image)))

			savemat(os.path.join(output_hypercube_directory, "%s.mat" % hs_filenumber), hypercube)

			imageio.imwrite(os.path.join(output_rgbn_directory, "%s_RGB.png" % hs_filenumber), rgb_image)
			imageio.imwrite(os.path.join(output_rgbn_directory, "%s_NIR.png" % hs_filenumber), nir_image)

			imageio.imwrite(os.path.join(output_secondary_rgbn_directory, "%s_RGB.png" % hs_filenumber), rgb_secondary_image)
			imageio.imwrite(os.path.join(output_secondary_rgbn_directory, "%s_NIR.png" % hs_filenumber), nir_image)

if __name__ == "__main__":
	main()