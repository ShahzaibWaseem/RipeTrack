import os
import sys
from saveMat import CommonLighting
sys.path.append("..")

import imageio
from PIL import Image
from glob import glob

import numpy as np
import pandas as pd

from utils import create_directory
from config import MOBILE_DATASET_DIR_NAME, TEST_ROOT_DATASET_DIR, VISUALIZATION_DIR_NAME, APPLICATION_NAME, SHELF_LIFE_GROUND_TRUTH_FILENAME, title_font_dict

import matplotlib.pyplot as plt

months = {"Aug": "08", "Sep": "09", "Mar": "03", "Apr": "04", "May": "05"}

def plotImages(image):
	fig, ax = plt.subplots(1, 1, figsize=(15, 5))
	ax.imshow(image)
	ax.title.set_text("RGB Image")
	plt.show()

def fixedAlign(image, aligningFactorX=35, aligningFactorY=82, mobileImageXShape=480, mobileImageYShape=640):
	""" Aligning the Mobile Images, Preset for front RGB and NIR cameras of Google Pixel 4XL. """
	image = image[aligningFactorY:aligningFactorY+mobileImageYShape, aligningFactorX:aligningFactorX+mobileImageXShape, :]
	return image

def viewImages(rgb_image, nir_image, rgb_image_aligned, plot_title, dataset_name, plot_filename):
	rgb_image_for = Image.fromarray(rgb_image_aligned)
	nir_image_for = Image.fromarray(nir_image)
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(35, 30))
	fig.suptitle(plot_title, **title_font_dict)
	ax1.imshow(rgb_image)
	ax2.imshow(nir_image)
	ax3.imshow(rgb_image_aligned)
	merged_image = Image.blend(rgb_image_for, nir_image_for, 0.5)
	ax4.imshow(merged_image)
	create_directory(os.path.join("..", TEST_ROOT_DATASET_DIR, APPLICATION_NAME, "%s_204ch" % dataset_name, VISUALIZATION_DIR_NAME))
	plt.savefig(os.path.join("..", TEST_ROOT_DATASET_DIR, APPLICATION_NAME, "%s_204ch" % dataset_name, VISUALIZATION_DIR_NAME, "%s.png" % plot_filename))
	plt.close()

def main():
	ground_truth_df = pd.read_csv(os.path.join(SHELF_LIFE_GROUND_TRUTH_FILENAME))
	check_filenames = []
	commonLighting = CommonLighting()

	for index in ground_truth_df.index:
		date, month = ground_truth_df["Date"][index].split("-")
		date = date.zfill(2)
		hs_filenames = [int(x) for x in ground_truth_df["HS Files"][index].split(",")]	# For HS Images
		# hs_filename = int(ground_truth_df["HS Files"][index])							# For Mobile Images
		mobile_fid = ground_truth_df["Fruit ID"][index]
		dataset_name = ground_truth_df["Fruit"][index].lower() + "-" + ground_truth_df["Type"][index].lower()
		fruit_name_short = ground_truth_df["Fruit"][index]
		fruit_name_short = fruit_name_short[0] + fruit_name_short[-1]

		mobile_dataset_input_directory = os.path.join("..", TEST_ROOT_DATASET_DIR, "mobile_%s_newnew" % APPLICATION_NAME, "rawImages")
		mobile_dataset_output_directory = os.path.join("..", TEST_ROOT_DATASET_DIR, APPLICATION_NAME, "%s_204ch" % dataset_name, MOBILE_DATASET_DIR_NAME)

		create_directory(mobile_dataset_output_directory)

		# IMG_2024_04_13_17_44_00_783_RGB(Pr_ID_0)[H].png, 'IMG_2024_04_16_12_09_03_138_NIR(Ne_ID_7)[H].png'
		mobile_rgbfilename = "IMG_2024_%s_%s_*_RGB(%s_ID_%s)*.png" % (months[month], date, fruit_name_short, mobile_fid)
		mobile_nirfilename = "IMG_2024_%s_%s_*_NIR(%s_ID_%s)*.png" % (months[month], date, fruit_name_short, mobile_fid)
		check_number_of_files_per_day = len(glob(os.path.join(mobile_dataset_input_directory, "IMG_2024_%s_%s_*(%s_ID_*)*.png" % (months[month], date, fruit_name_short))))

		# Checks if the files captured on Mobile have correct names. Soft Assertion/Warning
		rgb_glob_len = len(glob(os.path.join(mobile_dataset_input_directory, mobile_rgbfilename)))
		nir_glob_len = len(glob(os.path.join(mobile_dataset_input_directory, mobile_nirfilename)))
		if (rgb_glob_len != 2 or nir_glob_len != 2):
			print("\t\t\tCheck file numbers for Date %s-%s, FID: %s, Fruit Short Name: %s" % (date, month, mobile_fid, fruit_name_short))
			check_filenames.append("Date %s-%s, FID: %s, Fruit Short Name: %s" % (date, month, mobile_fid, fruit_name_short))

		for hs_filename, mobile_rgb, mobile_nir in zip(hs_filenames, sorted(glob(os.path.join(mobile_dataset_input_directory, mobile_rgbfilename))), sorted(glob(os.path.join(mobile_dataset_input_directory, mobile_nirfilename)))):
			assert os.path.split(mobile_rgb)[-1].split("_")[-1][-6] == os.path.split(mobile_nir)[-1].split("_")[-1][-6], "Chosen both Files from Different Lighting Settings"
			lighting = os.path.split(mobile_rgb)[-1].split("_")[-1][-6]
			if lighting != "H": continue
			rgb_image = imageio.imread(mobile_rgb)
			nir_image = imageio.imread(mobile_nir)
			rgb_image_aligned = fixedAlign(rgb_image)
			plot_title = "Date %s-%s, FID: %s, Fruit Short Name: %s" % (date, month, mobile_fid, fruit_name_short)
			# viewImages(rgb_image, nir_image, rgb_image_aligned, plot_title, dataset_name, hs_filename)

			print("Copied %s -> %s [Glob Len: %d]" % (os.path.split(mobile_rgb)[-1], "%s_RGB_%s.png" % (hs_filename, lighting), rgb_glob_len), "\tNumber of Files per day: %d\t" % check_number_of_files_per_day, "Fruit Name: %s\t" % fruit_name_short, "FID: %s\t" % mobile_fid, "Date: %s-%s" % (date, month))
			print("Copied %s -> %s [Glob Len: %d]" % (os.path.split(mobile_nir)[-1], "%s_NIR_%s.png" % (hs_filename, lighting), nir_glob_len), "\tNumber of Files per day: %d\t" % check_number_of_files_per_day, "Fruit Name: %s\t" % fruit_name_short, "FID: %s\t" % mobile_fid, "Date: %s-%s" % (date, month))

			nir_image = np.expand_dims(np.asarray(nir_image[:, :, 0]), axis=-1)
			daylight = commonLighting(rgb_image_aligned)

			imageio.imwrite(os.path.join(mobile_dataset_output_directory, "%s_RGB_%s.png" % (hs_filename, lighting)), rgb_image_aligned)
			imageio.imwrite(os.path.join(mobile_dataset_output_directory, "%s_RGB-D_%s.png" % (hs_filename, lighting)), daylight)
			imageio.imwrite(os.path.join(mobile_dataset_output_directory, "%s_NIR_%s.png" % (hs_filename, lighting)), nir_image)

	if len(check_filenames) != 0:
		print("Check These Filenames:", check_filenames)

if __name__ == "__main__":
	main()