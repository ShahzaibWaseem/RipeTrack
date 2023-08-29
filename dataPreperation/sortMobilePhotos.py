import os
import sys
sys.path.append("..")

import pandas as pd
from glob import glob
from shutil import copy2

from config import TEST_ROOT_DATASET_DIR, APPLICATION_NAME, SHELF_LIFE_GROUND_TRUTH_FILENAME, MOBILE_DATASET_DIR_NAME, create_directory

import matplotlib.pyplot as plt

months = {"Aug": "08", "Sep": "09"}

def plotImages(image):
	fig, ax = plt.subplots(1, 1, figsize=(15, 5))
	ax.imshow(image)
	ax.title.set_text("RGB Image")
	plt.show()

def fixedAlign(image1, image2):
	aligningFactorX = 37
	aligningFactorY = 87

def main():
	ground_truth_df = pd.read_csv(os.path.join(SHELF_LIFE_GROUND_TRUTH_FILENAME))

	for index in ground_truth_df.index:
		date, month = ground_truth_df["Date"][index].split("-")
		hs_filenames = [int(x) for x in ground_truth_df["HS Files"][index].split(",")]
		mobile_fid = ground_truth_df["Mobile FID"][index]
		dataset_name = ground_truth_df["Fruit"][index].lower() + "-" + ground_truth_df["Type"][index].replace("'", "").lower()
		fruit_name_short = ground_truth_df["Fruit"][index]
		fruit_name_short = fruit_name_short[0] + fruit_name_short[-1]

		mobile_dataset_input_directory = os.path.join("..", TEST_ROOT_DATASET_DIR, "mobile_%s" % APPLICATION_NAME)
		mobile_dataset_output_directory = os.path.join("..", TEST_ROOT_DATASET_DIR, APPLICATION_NAME, "%s_204ch" % dataset_name, MOBILE_DATASET_DIR_NAME)

		create_directory(mobile_dataset_output_directory)

		# IMG_2023_08_14_16_23_51_361_RGB_(Pr_ID_1).png
		mobile_rgbfilename = "IMG_2023_%s_%s_*_RGB_(%s_ID_%s).png" % (months[month], date, fruit_name_short, mobile_fid)
		mobile_nirfilename = "IMG_2023_%s_%s_*_NIR_(%s_ID_%s).png" % (months[month], date, fruit_name_short, mobile_fid)
		for hs_filename, mobile_rgb, mobile_nir in zip(hs_filenames, glob(os.path.join(mobile_dataset_input_directory, mobile_rgbfilename)), glob(os.path.join(mobile_dataset_input_directory, mobile_nirfilename))):
			copy2(mobile_rgb, os.path.join(mobile_dataset_output_directory, "%s_RGB.png" % hs_filename))
			copy2(mobile_nir, os.path.join(mobile_dataset_output_directory, "%s_NIR.png" % hs_filename))
			print("Copied %s -> %s" % (mobile_rgb.split("/")[-1], "%s_RGB.png" % hs_filename))
			print("Copied %s -> %s" % (mobile_nir.split("/")[-1], "%s_NIR.png" % hs_filename))

if __name__ == "__main__":
	main()