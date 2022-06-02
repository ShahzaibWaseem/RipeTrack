from __future__ import division

import os
import time
import numpy as np
from glob import glob

import torch
from torchsummary import summary

from models.model import Network
from models.resblock import ResNeXtBottleneck

from dataset import read_image
from utils import save_matv73, reconstruction, load_mat, initialize_logger
from loss import test_mrae, test_rrmse, test_msam, test_sid, test_psnr, test_ssim
from config import ILLUMINATIONS, TEST_ROOT_DATASET_DIR, TEST_DATASETS, MODEL_PATH, MODEL_NAME, BAND_SPACING, RGBN_BANDS, OUTPUT_BANDS, model_run_title, checkpoint_file, init_directories

def calculate_metrics(img_pred, img_gt):
	mrae = test_mrae(img_pred, img_gt)
	rrmse = test_rrmse(img_pred, img_gt)
	msam = test_msam(img_pred, img_gt)
	sid = test_sid(img_pred, img_gt)
	psnr = test_psnr(img_pred, img_gt)
	ssim = test_ssim(img_pred, img_gt)
	return mrae, rrmse, msam, sid, psnr, ssim

def inference(model, rgbn_from_cube=True):
	logger = initialize_logger(filename="test.log")
	log_string = "[%15s] Time: %0.9f, MRAE: %0.9f, RRMSE: %0.9f, SAM: %0.9f, SID: %0.9f, PSNR: %0.9f, SSIM: %0.9f"

	for test_dataset in TEST_DATASETS:
		for illumination in ILLUMINATIONS:
			print("\n" + model_run_title)
			logger.info(model_run_title)
			print("\nDataset: %s\nIllumination: %s\nModel: %s\n" % (test_dataset, illumination, checkpoint_file))
			logger.info("Dataset: %s\tIllumination: %s\tModel: %s\n" % (test_dataset, illumination, checkpoint_file))

			TEST_DATASET_DIR = os.path.join(TEST_ROOT_DATASET_DIR, "working_%s" % test_dataset)
			# GT_PATH =
			IMG_PATH = os.path.join(TEST_DATASET_DIR, "image") if not rgbn_from_cube else None
			INF_PATH = os.path.join(TEST_DATASET_DIR, "inference")

			for mat_filepath in sorted(glob(os.path.join(TEST_DATASET_DIR, "working_organic_204ch", "*.mat")) + glob(os.path.join(TEST_DATASET_DIR, "working_nonorganic_204ch", "*.mat"))):
				start_time = time.time()
				label = mat_filepath.split("/")[-2]

				hypercube = load_mat(mat_filepath)
				mat_filename = "_".join(mat_filepath.split("/")[-1].split("_")[0:2]) if illumination == "cfl_led" \
							   else mat_filepath.split("/")[-1].split("_")[0]

				if not rgbn_from_cube:
					rgb_filename = mat_filename + ".png"
					nir_filename = rgb_filename.replace("RGB", "NIR")

					image = read_image(rgb_filename, nir_filename)
				else:
					image = hypercube[:, :, RGBN_BANDS]
					image = np.transpose(image, [2, 0, 1])
				image = np.expand_dims(image, axis=0)

				hypercube = hypercube[:, :, ::BAND_SPACING]

				hypercube_pred = (reconstruction(image, model) + np.flip(reconstruction(np.flip(image, 2).copy(), model), 1))
				end_time = time.time() - start_time

				inf_mat_name = os.path.join(TEST_DATASET_DIR, label, "inference", "inf_" + mat_filename)
				save_matv73(inf_mat_name, hypercube_pred)

				mrae, rrmse, msam, sid, psnr, ssim = calculate_metrics(hypercube_pred+1, hypercube+1)

				print(log_string % (mat_filename, end_time, mrae, rrmse, msam, sid, psnr, ssim))
				logger.info(log_string % (mat_filename, end_time, mrae, rrmse, msam, sid, psnr, ssim))

def main():
	save_point = torch.load(os.path.join(MODEL_PATH, checkpoint_file))
	model_param = save_point["state_dict"]
	model = Network(ResNeXtBottleneck, block_num=10, input_channel=4, n_hidden=64, output_channel=OUTPUT_BANDS)
	model.load_state_dict(model_param)
	model = model.cuda()
	model.eval()
	print(summary(model=model, input_data=(4, 512, 512)))
	inference(model, rgbn_from_cube=True)

if __name__ == "__main__":
	# init_directories()
	main()