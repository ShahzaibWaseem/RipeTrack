import os
import json
import numpy as np
from glob import glob

from utils import load_mat
from loss import spectral_angle, spectral_divergence, test_mrae, test_rrmse, test_msam, test_sid, test_psnr, test_ssim

import matplotlib
import matplotlib.pyplot as plt

from config import ILLUMINATIONS, TEST_ROOT_DATASET_DIR, TEST_DATASETS, LOGS_PATH, var_name, text_font_dict, title_font_dict, plt_dict, fusion_techniques

def plotErrorAcrossBands():
	errors = {}
	for fusion in fusion_techniques:
		for test_dataset in TEST_DATASETS:
			for illumination in ILLUMINATIONS:
				TEST_DATASET_DIR = os.path.join(TEST_ROOT_DATASET_DIR, "working_%s" % test_dataset, "%s_%s_204ch" % (test_dataset, illumination), "test")
				
				GT_PATH = os.path.join(TEST_DATASET_DIR, "mat")
				INF_PATH = os.path.join(TEST_DATASET_DIR, "inference")

				print("\nFusion: %s\nDataset: %s\nIllumination: %s\n" % (fusion, test_dataset, illumination))

				for filename in glob(os.path.join(INF_PATH, fusion, "*.mat")):
					mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = [], [], [], [], [], []
					if(illumination == "cfl_led"):
						gt_filename = "_".join(filename.split("/")[-1].split(".")[0].split("_")[1:3])
					else:
						gt_filename = filename.split("/")[-1].split(".")[0].split("_")[-1]


					inf_file = load_mat(filename, var_name)[var_name]
					gt_file = load_mat(os.path.join(GT_PATH, gt_filename + ".mat"), var_name)[var_name]
					gt_file = gt_file[:,:,1:204:4]
					print(os.path.join(GT_PATH, gt_filename + ".mat"))


					for band in range(51):
						mrae_errors.append(float(test_mrae(inf_file[:,:, band], gt_file[:,:, band])))
						rrmse_errors.append(float(test_rrmse(inf_file[:,:, band], gt_file[:,:, band])))
						sam_errors.append(float(spectral_angle(inf_file[:,:, band].reshape(-1,)/4095, gt_file[:,:, band].reshape(-1,)/4095)))
						sid_errors.append(float(spectral_divergence(inf_file[:,:, band].reshape(-1,)/4095, gt_file[:,:, band].reshape(-1,)/4095)))
						psnr_errors.append(float(test_psnr(inf_file[:,:, band], gt_file[:,:, band])))
						ssim_errors.append(float(test_ssim(inf_file[:,:, band], gt_file[:,:, band])))

					errors.update({fusion + "_" + test_dataset + "_" + gt_filename: {"MRAE": mrae_errors, "RRMSE": rrmse_errors, "SAM": sam_errors, "SID": sid_errors, "PSNR": psnr_errors, "SSIM": ssim_errors}})
	return errors

def meanErrors(errors):
	mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
	for key, value in errors.items():
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
	return np.mean(mrae_errors, axis=1), np.mean(rrmse_errors, axis=1), np.mean(sam_errors, axis=1), np.mean(sid_errors, axis=1), np.mean(psnr_errors, axis=1), np.mean(ssim_errors, axis=1)

def readDataFromFile():
	errors = json.load(open(os.path.join(LOGS_PATH, "errors.json"), "r"))

	mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
	for key, value in errors.items():
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

if __name__ == "__main__":
	# errors = plotErrorAcrossBands()
	# mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = meanErrors(errors)
	# errors = {}
	# errors.update({"Object": {"MRAE": mrae_errors.tolist(), "RRMSE": rrmse_errors.tolist(), "SAM": sam_errors.tolist(), "SID": sid_errors.tolist(), "PSNR": psnr_errors.tolist(), "SSIM": ssim_errors.tolist()}})
	mrae_errors, rrmse_errors, sam_errors, sid_errors, psnr_errors, ssim_errors = readDataFromFile()
	plt.rcParams.update(plt_dict)
	matplotlib.rc("font", **text_font_dict)
	plt.rcParams["axes.grid"] = True
	plt.rcParams["axes.spines.right"] = False
	plt.rcParams["axes.spines.top"] = False
	fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
	axs[0][0].plot(mrae_errors)
	axs[0][0].set_ylabel("MRAE")
	axs[0][0].set_xlabel("Wavelength (nm)")
	axs[0][0].set_xticklabels(range(0, 1000, 200))
	axs[0][0].set_ylim(0, 0.3)

	axs[0][1].plot(rrmse_errors)
	axs[0][1].set_ylabel("RRMSE")
	axs[0][1].set_xlabel("Wavelength (nm)")
	axs[0][1].set_xticklabels(range(0, 1000, 200))
	axs[0][1].set_ylim(0, 0.3)

	axs[1][0].plot(sam_errors)
	axs[1][0].set_ylabel("SAM")
	axs[1][0].set_xlabel("Wavelength (nm)")
	axs[1][0].set_xticklabels(range(0, 1000, 200))
	axs[1][0].set_ylim(0, 0.3)

	axs[1][1].plot(sid_errors)
	axs[1][1].set_ylabel("SID")
	axs[1][1].set_xlabel("Wavelength (nm)")
	axs[1][1].set_xticklabels(range(0, 1000, 200))
	axs[1][1].set_ylim(0, 0.08)

	axs[2][0].plot(psnr_errors)
	axs[2][0].set_ylabel("PSNR")
	axs[2][0].set_xlabel("Wavelength (nm)")
	axs[2][0].set_xticklabels(range(0, 1000, 200))
	axs[2][0].set_ylim(25, 50)

	axs[2][1].plot(ssim_errors)
	axs[2][1].set_ylabel("SSIM")
	axs[2][1].set_xlabel("Wavelength (nm)")
	axs[2][1].set_xticklabels(range(0, 1000, 200))
	axs[2][1].set_ylim(0.8, 1)

	plt.tight_layout()
	plt.savefig(os.path.join(LOGS_PATH, "errors.png"))
	# print(mrae_errors, psnr_errors)
	# print(errors)
	# jsonFile = open(os.path.join(LOGS_PATH, "errors.json"), "w")
	# jsonFile.write(json.dumps(errors, indent=4))
	# jsonFile.close()