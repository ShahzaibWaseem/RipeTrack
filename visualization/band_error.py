import os, sys
sys.path.append(os.path.join(".."))

import json
import numpy as np
from glob import glob

from utils import load_mat, average
from loss import test_mrae, test_rrmse, spectral_angle, spectral_divergence, test_psnr, test_ssim

import matplotlib
import matplotlib.pyplot as plt

from config import BAND_SPACING, ILLUMINATIONS, MODEL_NAME, NORMALIZATION_FACTOR, TEST_ROOT_DATASET_DIR, TEST_DATASETS, LOGS_PATH, text_font_dict, plt_dict

def getBandErrors():
	""" returns a dictionary containing band wise errors for all evaluated results eg: {'avocado_1111': {}} """
	errors = {}
	log_string = "[%7s.mat] MRAE=%0.9f, RRMSE=%0.9f, SAM=%0.9f, SID=%0.9f, PSNR=%0.9f, SSIM=%0.9f"

	for test_dataset in TEST_DATASETS:
		for illumination in ILLUMINATIONS:
			TEST_DATASET_DIR = os.path.join(TEST_ROOT_DATASET_DIR, "working_%s" % test_dataset, "%s_%s_204ch" % (test_dataset, illumination), "test")
			GT_PATH = os.path.join(TEST_DATASET_DIR, "mat")
			INF_PATH = os.path.join("..", "AWAN", "test", "inference", "AWAN_4to51_%s_%s" % (test_dataset, illumination))

			print("\nDataset: %s\nIllumination: %s\n" % (test_dataset, illumination))

			for filename in sorted(glob(os.path.join(INF_PATH, "*.mat"))):
				mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = [], [], [], [], [], []
				if(illumination == "cfl_led"):
					gt_filename = "_".join(filename.split("/")[-1].split(".")[0].split("_")[1:3])
				else:
					gt_filename = filename.split("/")[-1].split(".")[0].split("_")[-1]

				inf_file = load_mat(filename)
				gt_file = load_mat(os.path.join(GT_PATH, gt_filename + ".mat"))
				gt_file = gt_file[:,:, ::BAND_SPACING]

				for band in range(51):
					mrae_errors.append(float(test_mrae(inf_file[:,:, band], gt_file[:,:, band])))
					rrmse_errors.append(float(test_rrmse(inf_file[:,:, band], gt_file[:,:, band])))
					sam_errors.append(float(spectral_angle(inf_file[:,:, band].reshape(-1,)/NORMALIZATION_FACTOR, gt_file[:,:, band].reshape(-1,)/NORMALIZATION_FACTOR)))
					sid_errors.append(float(spectral_divergence(inf_file[:,:, band].reshape(-1,)/NORMALIZATION_FACTOR, gt_file[:,:, band].reshape(-1,)/NORMALIZATION_FACTOR)))
					psnr_errors.append(float(test_psnr(inf_file[:,:, band], gt_file[:,:, band])))
					ssim_errors.append(float(test_ssim(inf_file[:,:, band], gt_file[:,:, band])))

				print(log_string % (gt_filename, average(mrae_errors), average(rrmse_errors), average(sam_errors), average(sid_errors), average(psnr_errors), average(ssim_errors)))
				errors.update({test_dataset + "_" + gt_filename: {"MRAE": mrae_errors, "RRMSE": rrmse_errors, "SAM": sam_errors, "SID": sid_errors, "PSNR": psnr_errors, "SSIM": ssim_errors}})
	return errors

def populateNumpyArrays(errors):
	""" creates numpy arrays with a dictionary (errors) as input """
	mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
	for _, value in errors.items():
		if mrae_errors.size == 0:
			mrae_errors = np.array(value["MRAE"]).reshape(-1, 1)
			rrmse_errors = np.array(value["RRMSE"]).reshape(-1, 1)
			sam_errors = np.array(value["SAM"]).reshape(-1, 1)
			sid_errors = np.array(value["SID"]).reshape(-1, 1)
			psnr_errors = np.array(value["PSNR"]).reshape(-1, 1)
			ssim_errors = np.array(value["SSIM"]).reshape(-1, 1)
		else:
			mrae_errors = np.append(mrae_errors, np.array(value["MRAE"]).reshape(-1, 1), axis=1)
			rrmse_errors = np.append(rrmse_errors, np.array(value["RRMSE"]).reshape(-1, 1), axis=1)
			sam_errors = np.append(sam_errors, np.array(value["SAM"]).reshape(-1, 1), axis=1)
			sid_errors = np.append(sid_errors, np.array(value["SID"]).reshape(-1, 1), axis=1)
			psnr_errors = np.append(psnr_errors, np.array(value["PSNR"]).reshape(-1, 1), axis=1)
			ssim_errors = np.append(ssim_errors, np.array(value["SSIM"]).reshape(-1, 1), axis=1)
	return mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors

def meanErrors(errors):
	""" takes the mean across the bands for all of the error metrics """
	mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = populateNumpyArrays(errors)
	return np.mean(mrae_errors, axis=1), np.mean(rrmse_errors, axis=1), np.mean(sam_errors, axis=1), np.mean(sid_errors, axis=1), np.mean(psnr_errors, axis=1), np.mean(ssim_errors, axis=1)

def readDataFromFile(json_file):
	""" reads dictionary form a json file """
	errors = json.load(open(os.path.join(LOGS_PATH, json_file), "r"))
	mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = populateNumpyArrays(errors)
	return mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors

def plotSingleMetric(error, row, ax, ylim, ylabel, xlabel="Wavelength (nm)", legend_loc="upper right"):
	""" plots error metrics on axis """
	x=range(400, 1001, 12)
	xlim=[400, 1000]

	ax.plot(x, error[:, row + 0], "k-", linewidth=2, label="HSCNN")
	ax.plot(x, error[:, row + 2], "b-", linewidth=2, label="AWAN")
	ax.plot(x, error[:, row + 4], "r-", linewidth=2, label="MobiSpectral")
	ax.legend(loc=legend_loc)
	ax.set_ylabel(ylabel)
	ax.set_xlabel(xlabel)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

def plotBandErrors(mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors, filename="errors.pdf"):
	""" creates a matplotlib for all error metrics """
	plt.rcParams.update(plt_dict)
	matplotlib.rc("font", **text_font_dict)
	plt.rcParams["axes.grid"] = True
	plt.rcParams["axes.spines.right"] = False
	plt.rcParams["axes.spines.top"] = False
	rows = len(["Meat", "Fruit"])

	_, axs = plt.subplots(nrows=rows, ncols=6, figsize=(45, 15))

	for row in range(rows):
		plotSingleMetric(mrae_errors, row, axs[row][0], ylim=[0, 0.5], ylabel="MRAE")
		plotSingleMetric(rrmse_errors, row, axs[row][1], ylim=[0, 0.4], ylabel="RRMSE")
		plotSingleMetric(sam_errors, row, axs[row][2], ylim=[0, 0.5], ylabel="SAM")
		plotSingleMetric(sid_errors, row, axs[row][3], ylim=[0, 0.2], ylabel="SID")
		plotSingleMetric(psnr_errors, row, axs[row][4], ylim=[20, 65], ylabel="PSNR", legend_loc="upper center")
		plotSingleMetric(ssim_errors, row, axs[row][5], ylim=[0.8, 1], ylabel="SSIM", legend_loc="lower right")
		axs[row][0].text(200, 0.205, "Meat" if row == 0 else "Fruit", rotation=90, weight="bold", size=36)

	plt.tight_layout()
	plt.savefig(os.path.join(LOGS_PATH, filename), bbox_inches="tight")

if __name__ == "__main__":
	os.chdir("..")

	# errors = getBandErrors()
	# print(errors.keys())
	# mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = meanErrors(errors)

	# errors = {}
	# errors.update({"AWAN - Fruit - CFL-LED": {"MRAE": mrae_errors.tolist(), "RRMSE": rrmse_errors.tolist(), "SAM": sam_errors.tolist(), "SID": sid_errors.tolist(), "PSNR": psnr_errors.tolist(), "SSIM": ssim_errors.tolist()}})

	# jsonFile = open(os.path.join(LOGS_PATH, "errors_fruit.json"), "w")
	# jsonFile.write(json.dumps(errors, indent=4))
	# jsonFile.close()

	mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = readDataFromFile(json_file="errors.json")
	plotBandErrors(mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors)