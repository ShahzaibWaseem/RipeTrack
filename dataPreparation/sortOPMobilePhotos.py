import os
import sys
from saveMat import CommonLighting
from sortMobilePhotos import fixedAlign, viewImages
sys.path.append("..")

import cv2
import imageio
from glob import glob

import numpy as np
import pandas as pd

from utils import create_directory
from config import OP_MOBILE_DATASET_DIR_NAME, TEST_ROOT_DATASET_DIR, APPLICATION_NAME, SHELF_LIFE_GROUND_TRUTH_FILENAME

months = {"Aug": "08", "Sep": "09", "Mar": "03", "Apr": "04", "May": "05"}

def main():
	ground_truth_df = pd.read_csv(os.path.join("ShelfLifeGroundTruth(NewNew).csv"))
	check_filenames, days_processed = [], []
	commonLighting = CommonLighting()

	for index in ground_truth_df.index:
		day = ground_truth_df["Day"][index]
		hs_filenames = [int(filenames.split(",")[0]) for filenames in ground_truth_df.loc[ground_truth_df["Day"] == day]["HS Files"].values]
		mobile_fids = [int(fids) for fids in ground_truth_df.loc[ground_truth_df["Day"] == day]["Fruit ID"].values]
		dataset_names = ["%s-%s" % (fruits.lower(), types.lower()) for fruits, types in ground_truth_df.loc[ground_truth_df["Day"] == day][["Fruit", "Type"]].values]
		date_combined = ["%s" % (date) for date in ground_truth_df.loc[ground_truth_df["Day"] == day]["Date"].values]
		date, month = date_combined[0].split("-")
		date = date.zfill(2)
		fruit_name_short = ground_truth_df["Fruit"][index]
		fruit_name_short = fruit_name_short[0] + fruit_name_short[-1]

		mobile_dataset_input_directory = os.path.join("..", TEST_ROOT_DATASET_DIR, "mobile_%s_newnew" % APPLICATION_NAME, "OnePlus Data")

		# 1709672942769_RGB.jpg
		mobile_rgbfilepath = glob(os.path.join(mobile_dataset_input_directory, "ShelfLife (Day %d) Halogen" % (day), "*_RGB.jpg"))
		mobile_nirfilepath = glob(os.path.join(mobile_dataset_input_directory, "ShelfLife (Day %d) Halogen" % (day), "*_NIR.jpg"))
		check_number_of_files_per_day = len(mobile_rgbfilepath + mobile_nirfilepath)

		# Checks if the files captured on Mobile have correct names. Soft Assertion/Warning
		rgb_glob_len = len(mobile_rgbfilepath)
		nir_glob_len = len(mobile_nirfilepath)
		if int(day) not in days_processed:
			if (rgb_glob_len != 1 or nir_glob_len != 1):
				print("\t\t\tCheck file numbers for Date %s-%s, Fruit Short Name: %s" % (date, month, fruit_name_short))
				check_filenames.append("Date %s-%s, Fruit Short Name: %s" % (date, month, fruit_name_short))

			for dataset_name, mobile_fid, hs_filename, mobile_rgb, mobile_nir in zip(dataset_names, mobile_fids, hs_filenames, sorted(mobile_rgbfilepath), sorted(mobile_nirfilepath)):
				mobile_dataset_output_directory = os.path.join("..", TEST_ROOT_DATASET_DIR, APPLICATION_NAME + "2", "%s_204ch" % dataset_name, OP_MOBILE_DATASET_DIR_NAME)
				create_directory(mobile_dataset_output_directory)
				rgb_image = imageio.imread(mobile_rgb)
				nir_image = imageio.imread(mobile_nir)

				rgb_image = cv2.resize(rgb_image, (600, 800))
				nir_image = cv2.resize(nir_image, (600, 800))

				rgb_image_aligned = fixedAlign(rgb_image, aligningFactorX=0, aligningFactorY=0, mobileImageXShape=480, mobileImageYShape=640)
				nir_image_aligned = fixedAlign(nir_image, aligningFactorX=18, aligningFactorY=25, mobileImageXShape=480, mobileImageYShape=640)

				viewImages(nir_image, rgb_image_aligned, nir_image_aligned, "Dataset: %s Date: %s-%s FID: %d" % (dataset_name, date, month, mobile_fid), dataset_name, hs_filename)

				print("Copied %s -> %s [Glob Len: %d]" % (os.path.split(mobile_rgb)[-1], "%s_RGB.png" % hs_filename, rgb_glob_len), "\tNumber of Files per day: %d\t" % check_number_of_files_per_day, "Fruit Name: %s\t" % dataset_name, "FID: %s\t" % mobile_fid, "Date: %s-%s" % (date, month))
				print("Copied %s -> %s [Glob Len: %d]" % (os.path.split(mobile_nir)[-1], "%s_NIR.png" % hs_filename, nir_glob_len), "\tNumber of Files per day: %d\t" % check_number_of_files_per_day, "Fruit Name: %s\t" % dataset_name, "FID: %s\t" % mobile_fid, "Date: %s-%s" % (date, month))
				nir_image = np.expand_dims(np.asarray(nir_image[:,:,0]), axis=-1)
				daylight = commonLighting(rgb_image_aligned)

				imageio.imwrite(os.path.join(mobile_dataset_output_directory, "%s_RGB.png" % hs_filename), rgb_image_aligned)
				imageio.imwrite(os.path.join(mobile_dataset_output_directory, "%s_RGB-D.png" % hs_filename), daylight)
				imageio.imwrite(os.path.join(mobile_dataset_output_directory, "%s_NIR.png" % hs_filename), nir_image_aligned)
		days_processed.append(int(day))

if __name__ == "__main__":
	main()