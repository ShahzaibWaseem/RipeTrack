import os
import sys
sys.path.append(os.path.join(".."))

import json
import numpy as np
from glob import glob

from utils import load_mat, average
from loss import test_mrae, test_rrmse, test_msam, test_sid, test_psnr, test_ssim

from utils import create_directory
from config import GT_HYPERCUBES_DIR_NAME, RECONSTRUCTED_HS_DIR_NAME, GT_REMOVED_IR_CUTOFF_RECONSTRUCTED_DIR_NAME,\
	TRAIN_VAL_TEST_SPLIT_DIR_NAME, VISUALIZATION_DIR_NAME, TEST_ROOT_DATASET_DIR, TEST_DATASETS, LOGS_PATH,\
	EPS, BANDS, APPLICATION_NAME, text_font_dict, plt_dict

import matplotlib
import matplotlib.pyplot as plt

def calculate_metrics(img_pred, img_gt, expand=False):
	img_pred_flat = np.expand_dims(img_pred, axis=0) if expand else img_pred
	img_gt_flat = np.expand_dims(img_gt, axis=0) if expand else img_gt
	mrae = test_mrae(img_pred, img_gt)
	rrmse = test_rrmse(img_pred, img_gt)
	msam = test_msam(img_pred_flat, img_gt_flat)
	sid = test_sid(img_pred_flat, img_gt_flat)
	psnr = test_psnr(img_pred, img_gt, max_value=1)		# max_value = 1 for 0-1 normalized images
	ssim = test_ssim(img_pred, img_gt, max_value=1)		# max_value = 1 for 0-1 normalized images
	return mrae, rrmse, msam, sid, psnr, ssim

def getBandErrors():
	""" returns a dictionary containing band wise errors for all evaluated results eg: {'avocado_1111': {}} """
	errors = {}
	log_string = "[%15s] MRAE=%0.9f, RRMSE=%0.9f, SAM=%0.9f, SID=%0.9f, PSNR=%0.9f, SSIM=%0.9f"

	for dataset in TEST_DATASETS:
		directory = os.path.join(TEST_ROOT_DATASET_DIR, APPLICATION_NAME, "{}_204ch".format(dataset), GT_HYPERCUBES_DIR_NAME)
		inf_directory = os.path.join(directory, RECONSTRUCTED_HS_DIR_NAME)
		print(" " * 19, "{0:62}".format(inf_directory))

		with open(os.path.join(directory, TRAIN_VAL_TEST_SPLIT_DIR_NAME, "test.txt"), "r") as test_file:
			hypercube_list = [filename.replace("\n", ".mat") for filename in test_file]

		for filename in hypercube_list:
		# for filename in sorted(glob(os.path.join(directory, "*.mat"))):
			mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = [], [], [], [], [], []

			inf_hypercube = load_mat(os.path.join(inf_directory, filename))
			# inf_hypercube = (inf_hypercube - inf_hypercube.min()) / (inf_hypercube.max() - inf_hypercube.min())
			gt_hypercube = load_mat(os.path.join(directory, filename))
			gt_hypercube = gt_hypercube[:,:,BANDS]
			gt_hypercube = (gt_hypercube - gt_hypercube.min()) / (gt_hypercube.max() - gt_hypercube.min())
			# gt_hypercube = np.maximum(np.minimum(gt_hypercube, 1.0), 0.0)
			gt_hypercube = gt_hypercube + EPS

			for band in range(len(BANDS)):
				inf_band = inf_hypercube[:,:,band]
				gt_band = gt_hypercube[:,:,band]
				mrae, rrmse, msam, sid, psnr, ssim = calculate_metrics(inf_band, gt_band, expand=True)
				mrae_errors.append(mrae if not np.isinf(mrae) else 0)
				rrmse_errors.append(rrmse if not np.isinf(rrmse) else 0)
				sam_errors.append(msam)
				sid_errors.append(sid)
				psnr_errors.append(psnr)
				ssim_errors.append(ssim)

			print(log_string % (os.path.split(filename)[-1], average(mrae_errors), average(rrmse_errors), average(sam_errors), average(sid_errors), average(psnr_errors), average(ssim_errors)))
			errors.update({dataset + "_" + filename: {"MRAE": mrae_errors, "RRMSE": rrmse_errors, "SAM": sam_errors, "SID": sid_errors, "PSNR": psnr_errors, "SSIM": ssim_errors}})
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
	errors = json.load(open(os.path.join(VISUALIZATION_DIR_NAME, json_file), "r"))
	mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = populateNumpyArrays(errors)
	return mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors

def plotSingleMetric(error, ax, ylim, ylabel, xlabel="Wavelength (nm)", legend_loc="upper right", label="MobiSLP", color="r"):
	""" plots error metrics on axis """
	x=range(400, 1004, 9)
	xlim=[400, 1000]

	ax.plot(x, error, "r-", linewidth=5, label=label, color=color)
	# ax.legend(loc=legend_loc)
	ax.set_ylabel(ylabel)
	ax.set_xlabel(xlabel)
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

def plotBandErrors(mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors, filename="errors.pdf", save=True):
	""" creates a matplotlib for all error metrics """
	plt.rcParams.update(plt_dict)
	matplotlib.rc("font", **text_font_dict)
	plt.rcParams["axes.grid"] = True
	plt.rcParams["axes.spines.right"] = False
	plt.rcParams["axes.spines.top"] = False
	# rows = len(["Fruit"])

	fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 20))

	plotSingleMetric(mrae_errors, axs[0][0], ylim=[0, 0.5], ylabel="MRAE")
	plotSingleMetric(rrmse_errors, axs[0][1], ylim=[0, 0.4], ylabel="RRMSE")
	plotSingleMetric(sam_errors, axs[0][2], ylim=[0, 0.5], ylabel="SAM")
	plotSingleMetric(sid_errors, axs[1][0], ylim=[0, 0.2], ylabel="SID")
	plotSingleMetric(psnr_errors, axs[1][1], ylim=[20, 65], ylabel="PSNR", legend_loc="upper center")
	plotSingleMetric(ssim_errors, axs[1][2], ylim=[0.8, 1], ylabel="SSIM", legend_loc="lower right")
	plt.tight_layout()
	if save:
		plt.savefig(os.path.join(VISUALIZATION_DIR_NAME, filename), bbox_inches="tight")
	plt.close()


mrae_errors_combined, rrmse_errors_combined, sam_errors_combined, sid_errors_combined, psnr_errors_combined, ssim_errors_combined, titles = [], [], [], [], [], [], []

def combineAllArrays(mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors, title):
	global mrae_errors_combined, rrmse_errors_combined, sam_errors_combined, sid_errors_combined, psnr_errors_combined, ssim_errors_combined, titles
	mrae_errors_combined.append(mrae_errors)
	rrmse_errors_combined.append(rrmse_errors)
	sam_errors_combined.append(sam_errors)
	sid_errors_combined.append(sid_errors)
	psnr_errors_combined.append(psnr_errors)
	ssim_errors_combined.append(ssim_errors)
	titles.append(title)

def plotAllBandErrors(filename="errors.pdf"):
	global mrae_errors_combined, rrmse_errors_combined, sam_errors_combined, sid_errors_combined, psnr_errors_combined, ssim_errors_combined, titles

	plt.rcParams.update(plt_dict)
	matplotlib.rc("font", **text_font_dict)
	plt.rcParams["axes.grid"] = True
	plt.rcParams["axes.spines.right"] = False
	plt.rcParams["axes.spines.top"] = False
	# rows = len(["Fruit"])

	fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 20))
	colors = ["r", "g", "b", "c", "m", "y", "k", "w"]
	
	for ix, title in enumerate(titles):
		plotSingleMetric(mrae_errors_combined[ix], axs[0][0], ylim=[0, 0.7], ylabel="MRAE", label=title, color=colors[ix])
		plotSingleMetric(rrmse_errors_combined[ix], axs[0][1], ylim=[0, 0.4], ylabel="RRMSE", label=title, color=colors[ix])
		plotSingleMetric(sam_errors_combined[ix], axs[0][2], ylim=[0, 0.4], ylabel="SAM", label=title, color=colors[ix])
		plotSingleMetric(sid_errors_combined[ix], axs[1][0], ylim=[0, 0.2], ylabel="SID", label=title, color=colors[ix])
		plotSingleMetric(psnr_errors_combined[ix], axs[1][1], ylim=[10, 45], ylabel="PSNR", legend_loc="upper right", label=title, color=colors[ix])
		plotSingleMetric(ssim_errors_combined[ix], axs[1][2], ylim=[0.8, 1], ylabel="SSIM", legend_loc="lower right", label=title, color=colors[ix])
	plt.tight_layout()
	plt.savefig(os.path.join(VISUALIZATION_DIR_NAME, "tempJSON", filename), bbox_inches="tight")
	plt.close()

def processRawJSON(jsonfile, title, fruitname="pear-williams"):
	global mrae_errors_combined, rrmse_errors_combined, sam_errors_combined, sid_errors_combined, psnr_errors_combined, ssim_errors_combined, titles
	errors = json.load(open(os.path.join(VISUALIZATION_DIR_NAME, "tempJSON", jsonfile), "r"))
	mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
	for key, value in errors.items():
		if fruitname in key:
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

	mrae_errors_combined.append(np.mean(mrae_errors, axis=1))
	rrmse_errors_combined.append(np.mean(rrmse_errors, axis=1))
	sam_errors_combined.append(np.mean(sam_errors, axis=1))
	sid_errors_combined.append(np.mean(sid_errors, axis=1))
	psnr_errors_combined.append(np.mean(psnr_errors, axis=1))
	ssim_errors_combined.append(np.mean(ssim_errors, axis=1))
	titles.append(title)
	print("Title: %s\tSTD DEV\tMRAE: %0.2f, RRMSE: %0.2f, SAM: %0.2f, SID: %0.2f, PSNR: %0.2f, SSIM: %0.2f" % (title, np.std(np.mean(mrae_errors, axis=1)), np.std(np.mean(rrmse_errors, axis=1)), np.std(np.mean(sam_errors, axis=1)), np.std(np.mean(sid_errors, axis=1)), np.std(np.mean(psnr_errors, axis=1)), np.std(np.mean(ssim_errors, axis=1))))

if __name__ == "__main__":
	os.chdir("..")
	create_directory(os.path.join(VISUALIZATION_DIR_NAME, "tempJSON"))

	# errors = getBandErrors()
	# print(errors.keys())
	# mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = meanErrors(errors)

	# jsonFile = open(os.path.join(VISUALIZATION_DIR_NAME, "tempJSON", "errorsRaw(RGBto68)[Vanilla].json"), "w")
	# jsonFile.write(json.dumps(str(errors), indent=4))
	# jsonFile.close()

	# errors = {}
	# errors.update({"MSLP": {"MRAE": mrae_errors.tolist(), "RRMSE": rrmse_errors.tolist(), "SAM": sam_errors.tolist(), "SID": sid_errors.tolist(), "PSNR": psnr_errors.tolist(), "SSIM": ssim_errors.tolist()}})

	# jsonFile = open(os.path.join(VISUALIZATION_DIR_NAME, "tempJSON", "errors(RGBto68)[Vanilla].json"), "w")
	# jsonFile.write(json.dumps(str(errors), indent=4))
	# jsonFile.close()

	# mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = readDataFromFile(json_file=os.path.join("tempJSON","errors(RGBNto68)[VanillaMRAE].json"))
	# plotBandErrors(mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors, filename=os.path.join("tempJSON", "errors(RGNto68)[VanillaMRAE].pdf"))
	fruitname = "pear-williams"
	processRawJSON("errorsRaw(RGBto68)[Vanilla].json", "Pear Bartlett", fruitname)
	# processRawJSON("errorsRaw(RGBto68)[Vanilla].json", "Avocado Organic", "avocado-organic")


	# processRawJSON("errorsRaw(RGBto68)[IR Cutoff Removed].json", "RGB [IR CR]", fruitname)
	# processRawJSON("errorsRaw(RGBNto68).json", "RGB+FaceID", fruitname)
	# processRawJSON("errorsRaw(RGBNto68)[IR Cutoff Removed].json", "Custom Sensor", fruitname)
	# processRawJSON("errorsRaw(RGBNto68)[Vanilla].json", "RGBN [Vanilla]", fruitname)
	# processRawJSON("errorsRaw(RGBNto68)[Losses + NIR Augmentation].json", "RGBN [L+A]", fruitname)

	# mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = readDataFromFile(json_file=os.path.join("tempJSON","errors(RGBto68).json"))
	# combineAllArrays(mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors, "RGB")
	# mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = readDataFromFile(json_file=os.path.join("tempJSON","errors(RGBto68)[IR Cutoff Removed].json"))
	# combineAllArrays(mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors, "RGB [IR CR]")
	# mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = readDataFromFile(json_file=os.path.join("tempJSON","errors(RGBNto68).json"))
	# combineAllArrays(mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors, "RGB+FaceID")
	# mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = readDataFromFile(json_file=os.path.join("tempJSON","errors(RGBNto68)[IR Cutoff Removed].json"))
	# combineAllArrays(mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors, "Custom Sensor")
	# mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = readDataFromFile(json_file=os.path.join("tempJSON","errors(RGBNto68)[Vanilla].json"))
	# combineAllArrays(mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors, "RGBN [Vanilla]")
	# mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = readDataFromFile(json_file=os.path.join("tempJSON","errors(RGBNto68)[Losses + NIR Augmentation].json"))
	# combineAllArrays(mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors, "RGBN [L+A]")

	plotAllBandErrors(filename="errors(%s).pdf" % ("VanillaMST"))