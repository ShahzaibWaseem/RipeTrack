from __future__ import division

import os
import time
import numpy as np
from glob import glob
from imageio import imread

import torch
from torchsummary import summary
from models.MST import MST_Plus_Plus

from utils import save_mat, load_mat, initialize_logger, visualize_gt_pred_hs_data, get_best_checkpoint
from loss import test_mrae, test_rrmse, test_msam, test_sid, test_psnr, test_ssim
from config import MODEL_PATH, TEST_ROOT_DATASET_DIR, TEST_DATASETS, APPLICATION_NAME, BANDS, MOBILE_DATASET_DIR_NAME, RECONSTRUCTED_HS_DIR_NAME, GT_RGBN_DIR_NAME, GT_SECONDARY_RGB_CAM_DIR_NAME, MOBILE_RECONSTRUCTED_HS_DIR_NAME, EPS, model_run_title, checkpoint_file, device, create_directory

def calculate_metrics(img_pred, img_gt):
	mrae = test_mrae(img_pred, img_gt)
	rrmse = test_rrmse(img_pred, img_gt)
	msam = test_msam(img_pred, img_gt)
	sid = test_sid(img_pred, img_gt)
	psnr = test_psnr(img_pred, img_gt, max_p=1)		# max_p = 1 for 0-1 normalized images
	ssim = test_ssim(img_pred, img_gt, max_p=1)		# max_p = 1 for 0-1 normalized images
	return mrae, rrmse, msam, sid, psnr, ssim

def inference(model, checkpoint_filename, mobile_reconstruction=False):
	# input_transform, label_transform = get_required_transforms(task="reconstruction")
	logger = initialize_logger(filename="test.log")
	log_string = "[%15s] Time: %0.9f, MRAE: %0.9f, RRMSE: %0.9f, SAM: %0.9f, SID: %0.9f, PSNR: %0.9f, SSIM: %0.9f"
	TEST_DATASET_DIR = os.path.join(TEST_ROOT_DATASET_DIR, APPLICATION_NAME)

	for test_dataset in TEST_DATASETS:
		directory = os.path.join(TEST_DATASET_DIR, "%s_204ch" % test_dataset)
		OUT_PATH = os.path.join(directory, RECONSTRUCTED_HS_DIR_NAME) if not mobile_reconstruction else os.path.join(directory, MOBILE_RECONSTRUCTED_HS_DIR_NAME)
		create_directory(OUT_PATH)

		print("\n" + model_run_title)
		logger.info(model_run_title)

		if mobile_reconstruction:
			print("Mobile Reconstruction")
			logger.info("Mobile Reconstruction") if mobile_reconstruction else None

		print("Dataset: %s\nTest Directory: %s\nModel: %s\n" % (APPLICATION_NAME, test_dataset, checkpoint_filename))
		logger.info("Dataset: %s\tTest Directory: %s\tModel: %s\n" % (APPLICATION_NAME, test_dataset, checkpoint_filename))

		for mat_filepath in sorted(glob(os.path.join(directory, "*.mat"))):
			start_time = time.time()
			mat_filename = os.path.split(mat_filepath)[-1]

			hypercube = load_mat(mat_filepath)
			hypercube = hypercube[:, :, BANDS]
			hypercube = (hypercube - hypercube.min()) / (hypercube.max() - hypercube.min())
			hypercube = hypercube + EPS

			rgb_filename = mat_filename.replace(".mat", "_RGB.png")
			rgb_image = imread(os.path.join(directory, GT_RGBN_DIR_NAME if not mobile_reconstruction else MOBILE_DATASET_DIR_NAME, rgb_filename))
			rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
			rgb_image = np.transpose(rgb_image, [2, 0, 1])
			rgb_image = np.expand_dims(rgb_image, axis=0)

			nir_filename = mat_filename.replace(".mat", "_NIR.png")
			nir_image = imread(os.path.join(directory, GT_RGBN_DIR_NAME if not mobile_reconstruction else MOBILE_DATASET_DIR_NAME, nir_filename))
			nir_image = (nir_image - nir_image.min()) / (nir_image.max() - nir_image.min())
			nir_image = np.expand_dims(np.asarray(nir_image), axis=-1)
			nir_image = np.transpose(nir_image, [2, 0, 1])
			nir_image = np.expand_dims(nir_image, axis=0)

			image = torch.Tensor(np.concatenate((rgb_image, nir_image), axis=1)).float().to(device)

			with torch.no_grad():
				hypercube_pred = model(image)
			hypercube_pred = np.transpose(hypercube_pred.squeeze(0).cpu().detach().numpy(), [1, 2, 0])
			# hypercube_pred = hypercube_pred + EPS			# should work without this line but just in case

			end_time = time.time() - start_time
			hypercube_pred_filepath = os.path.join(OUT_PATH, mat_filename)
			save_mat(hypercube_pred_filepath, hypercube_pred)

			if not mobile_reconstruction:
				# visualize_gt_pred_hs_data(hypercube, hypercube_pred, 12)
				mrae, rrmse, msam, sid, psnr, ssim = calculate_metrics(hypercube_pred, hypercube)

				print(log_string % (mat_filename, end_time, mrae, rrmse, msam, sid, psnr, ssim))
				logger.info(log_string % (mat_filename, end_time, mrae, rrmse, msam, sid, psnr, ssim))
			else:
				print("[%15s] Time: %0.9f" % (mat_filename, end_time))
				logger.info("[%15s] Time: %0.9f" % (mat_filename, end_time))

def main():
	# checkpoint_filename, epoch, iter, model_param, optimizer, val_loss, val_acc = get_best_checkpoint(task="reconstruction")
	checkpoint_filename = checkpoint_file
	checkpoint = torch.load(os.path.join(MODEL_PATH, "reconstruction", checkpoint_file))
	model_param = checkpoint["state_dict"]
	model = MST_Plus_Plus(in_channels=4, out_channels=len(BANDS), n_feat=len(BANDS), stage=3)
	model.load_state_dict(model_param)
	model = model.to(device)
	model.eval()
	print(summary(model=model, input_data=(4, 512, 512)))
	inference(model, checkpoint_filename, mobile_reconstruction=False)

if __name__ == "__main__":
	main()