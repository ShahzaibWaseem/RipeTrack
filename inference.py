from __future__ import division

import os
import time
import numpy as np
import pandas as pd
from glob import glob
from imageio import imread

import torch
from torchsummary import summary
from models.MST import MST_Plus_Plus

from utils import save_mat, save_mat_patched, load_mat, load_mat_patched, initialize_logger, visualize_gt_pred_hs_data, get_best_checkpoint, AverageMeter
from loss import test_mrae, test_rrmse, test_msam, test_sid, test_psnr, test_ssim
from config import MODEL_PATH, TEST_ROOT_DATASET_DIR, TEST_DATASETS, APPLICATION_NAME, BANDS, TRAIN_VAL_TEST_SPLIT_DIR_NAME, CLASSIFICATION_PATCH_SIZE, STRIDE, DATA_PREP_PATH, GT_DATASET_CROPS_FILENAME, MOBILE_DATASET_CROPS_FILENAME, MOBILE_DATASET_DIR_NAME, RECONSTRUCTED_HS_DIR_NAME, GT_RGBN_DIR_NAME, GT_AUXILIARY_RGB_CAM_DIR_NAME, GT_REMOVED_IR_CUTOFF_DIR_NAME, MOBILE_RECONSTRUCTED_HS_DIR_NAME, PATCHED_INFERENCE, PATCHED_HS_DIR_NAME, EPS, var_name, model_run_title, checkpoint_file, device, create_directory

def calculate_metrics(img_pred, img_gt):
	mrae = test_mrae(img_pred, img_gt)
	rrmse = test_rrmse(img_pred, img_gt)
	msam = test_msam(img_pred, img_gt, max_value=1)
	sid = test_sid(img_pred, img_gt, max_value=1)
	psnr = test_psnr(img_pred, img_gt, max_value=1)
	ssim = test_ssim(img_pred, img_gt, max_value=1)
	return mrae, rrmse, msam, sid, psnr, ssim

def inference(model, checkpoint_filename, mobile_reconstruction=False, patched_inference=False):
	# input_transform, label_transform = get_required_transforms(task="reconstruction")
	logger = initialize_logger(filename="test.log")
	log_string = "[%15s] Time: %0.9f, MRAE: %0.9f, RRMSE: %0.9f, SAM: %0.9f, SID: %0.9f, PSNR: %0.9f, SSIM: %0.9f"
	log_string_avg = "%15s, %0.4f \pm %0.2f, %0.4f \pm %0.2f, %0.4f \pm %0.2f, %0.4f \pm %0.2f, %0.1f \pm %0.1f, %0.4f \pm %0.2f"
	log_string_avg_combined = "%15s, \\textbf{%0.4f \pm %0.2f}, \\textbf{%0.4f \pm %0.2f}, \\textbf{%0.4f \pm %0.2f}, \\textbf{%0.4f \pm %0.2f}, \\textbf{%0.1f \pm %0.1f}, \\textbf{%0.4f \pm %0.2f}"

	TEST_DATASET_DIR = os.path.join(TEST_ROOT_DATASET_DIR, APPLICATION_NAME)

	crops_filepath = os.path.join(DATA_PREP_PATH, MOBILE_DATASET_CROPS_FILENAME if mobile_reconstruction else GT_DATASET_CROPS_FILENAME)
	crops_df = pd.read_csv(crops_filepath)
	print("Using {} Crop File".format(crops_filepath))

	crops_df["w"] = crops_df["xmax"] - crops_df["xmin"]
	crops_df["h"] = crops_df["ymax"] - crops_df["ymin"]
	min_hc, max_hc = np.inf, -np.inf
	min_phc, max_phc = np.inf, -np.inf
	losses_mrae_combined = AverageMeter()
	losses_rmse_combined = AverageMeter()
	losses_sam_combined = AverageMeter()
	losses_sid_combined = AverageMeter()
	losses_psnr_combined = AverageMeter()
	losses_ssim_combined = AverageMeter()

	for test_dataset in TEST_DATASETS:
		losses_mrae = AverageMeter()
		losses_rmse = AverageMeter()
		losses_psnr = AverageMeter()
		losses_sam = AverageMeter()
		losses_sid = AverageMeter()
		losses_ssim = AverageMeter()
		directory = os.path.join(TEST_DATASET_DIR, "%s_204ch" % test_dataset)
		OUT_PATH = os.path.join(directory, RECONSTRUCTED_HS_DIR_NAME) if not mobile_reconstruction else os.path.join(directory, MOBILE_RECONSTRUCTED_HS_DIR_NAME)
		create_directory(OUT_PATH)
		create_directory(os.path.join(OUT_PATH, PATCHED_HS_DIR_NAME)) if patched_inference else None

		print("\n" + model_run_title)
		logger.info(model_run_title)

		if mobile_reconstruction:
			print("Mobile Reconstruction")
			logger.info("Mobile Reconstruction") if mobile_reconstruction else None

		print("Dataset: %s\nTest Directory: %s\nModel: %s\n" % (APPLICATION_NAME, test_dataset, checkpoint_filename))
		logger.info("Dataset: %s\tTest Directory: %s\tModel: %s\n" % (APPLICATION_NAME, test_dataset, checkpoint_filename))
		p_mrae, p_rrmse, p_msam, p_sid, p_psnr, p_ssim, num_patches = 0, 0, 0, 0, 0, 0, 0

		with open(os.path.join(TEST_DATASET_DIR, "%s_204ch" % test_dataset, TRAIN_VAL_TEST_SPLIT_DIR_NAME, "test.txt"), "r") as test_file:
			hypercube_list = [filename.replace("\n", ".mat") for filename in test_file]

		for mat_filepath in hypercube_list:
		# for mat_filepath in sorted(glob(os.path.join(directory, "*.mat"))):
			start_time = time.time()
			mat_filename = os.path.split(mat_filepath)[-1]
			mat_number = mat_filename.split("_")[0].split(".")[0]

			hypercube = load_mat(os.path.join(directory, mat_filepath))

			# hypercube = load_mat(os.path.join(mat_filepath))
			hypercube = hypercube[:, :, BANDS]
			hypercube = (hypercube - hypercube.min()) / (hypercube.max() - hypercube.min())
			min_hc, max_hc = min(min_hc, hypercube.min()), max(max_hc, hypercube.max())
			# hypercube = np.maximum(np.minimum(hypercube, 1.0), 0.0)
			hypercube = hypercube + EPS

			rgb_filename = mat_filename.replace(".mat", "_RGB%s.png" % "-D")
			rgb_image = np.float32(imread(os.path.join(directory, GT_RGBN_DIR_NAME if not mobile_reconstruction else MOBILE_DATASET_DIR_NAME, rgb_filename)))
			rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

			nir_filename = mat_filename.replace(".mat", "_NIR.png")
			nir_image = np.float32(imread(os.path.join(directory, GT_RGBN_DIR_NAME if not mobile_reconstruction else MOBILE_DATASET_DIR_NAME, nir_filename)))
			nir_image = (nir_image - nir_image.min()) / (nir_image.max() - nir_image.min())
			nir_image = np.expand_dims(np.asarray(nir_image), axis=-1)

			image = np.dstack((rgb_image, nir_image))
			# image = rgb_image
			image = np.transpose(image, [2, 0, 1])
			image = np.expand_dims(image, axis=0)
			image_tensor = torch.Tensor(image).float().to(device)

			with torch.no_grad():
				hypercube_pred = model(image_tensor)

			hypercube_pred = np.transpose(hypercube_pred.squeeze(0).cpu().detach().numpy(), [1, 2, 0])
			# hypercube_pred = hypercube_pred + EPS			# should work without this line but just in case
			hypercube_pred = np.maximum(np.minimum(hypercube_pred, 1.0), 0.0)

			min_phc, max_phc = min(min_phc, hypercube_pred.min()), max(max_phc, hypercube_pred.max())
			# print("HC Min: {}, Max: {}\tPred Min: {}, Max: {}".format(min_hc, max_hc, min_phc, max_phc))

			end_time = time.time() - start_time
			hypercube_pred_filepath = os.path.join(OUT_PATH, mat_filename)
			save_mat(hypercube_pred_filepath, hypercube_pred)

			if not mobile_reconstruction:
				# visualize_gt_pred_hs_data(hypercube, hypercube_pred, 12)
				mrae, rrmse, msam, sid, psnr, ssim = calculate_metrics(hypercube_pred, hypercube)
				losses_mrae.update(mrae)
				losses_rmse.update(rrmse)
				losses_sam.update(msam)
				losses_sid.update(sid)
				losses_psnr.update(psnr)
				losses_ssim.update(ssim)

				losses_mrae_combined.update(mrae)
				losses_rmse_combined.update(rrmse)
				losses_sam_combined.update(msam)
				losses_sid_combined.update(sid)
				losses_psnr_combined.update(psnr)
				losses_ssim_combined.update(ssim)

				print(log_string % (mat_filename, end_time, mrae, rrmse, msam, sid, psnr, ssim))
				logger.info(log_string % (mat_filename, end_time, mrae, rrmse, msam, sid, psnr, ssim))
			else:
				print("[%15s] Time: %0.9f" % (mat_filename, end_time))
				logger.info("[%15s] Time: %0.9f" % (mat_filename, end_time))

			if patched_inference:
				crop_record = crops_df[crops_df["image"].isin(["{}_RGB.png".format(mat_number)])]
				hypercube_combined = {}
				xmin = int(crop_record["xmin"].iloc[0])
				ymin = int(crop_record["ymin"].iloc[0])
				xmax = int(crop_record["xmax"].iloc[0])
				ymax = int(crop_record["ymax"].iloc[0])
				# hypercube_joined = hypercube.copy()

				for patch_i in range(xmin, xmax, CLASSIFICATION_PATCH_SIZE):
					if patch_i+CLASSIFICATION_PATCH_SIZE > xmax: continue
					for patch_j in range(ymin, ymax, CLASSIFICATION_PATCH_SIZE):
						if patch_j+CLASSIFICATION_PATCH_SIZE > ymax: continue
						imageCrop = image[:, :, patch_i:patch_i+CLASSIFICATION_PATCH_SIZE, patch_j:patch_j+CLASSIFICATION_PATCH_SIZE]
						hypercubeCrop = hypercube[patch_i:patch_i+CLASSIFICATION_PATCH_SIZE, patch_j:patch_j+CLASSIFICATION_PATCH_SIZE, :]
						# hypercubeCrop = np.transpose(hypercubeCrop, [1, 2, 0])
						if (imageCrop.shape[2:4] != (CLASSIFICATION_PATCH_SIZE, CLASSIFICATION_PATCH_SIZE)):
							# hypercube_joined[patch_i:patch_i+CLASSIFICATION_PATCH_SIZE, patch_j:patch_j+CLASSIFICATION_PATCH_SIZE, :] = 0
							continue
						image_tensor = torch.Tensor(imageCrop).float().to(device)
						with torch.no_grad():
							hypercube_pred = model(image_tensor)
						# hypercube_pred = hypercube_pred.squeeze(0).cpu().detach().numpy()
						hypercube_pred = np.transpose(hypercube_pred.squeeze(0).cpu().detach().numpy(), [1, 2, 0]) + EPS
						if not mobile_reconstruction:
							mrae, rrmse, msam, sid, psnr, ssim = calculate_metrics(hypercube_pred + EPS, hypercubeCrop)
							p_mrae += mrae
							p_rrmse += rrmse
							p_msam += msam
							p_sid += sid
							p_psnr += psnr
							p_ssim += ssim
							num_patches += 1

						# hypercube_joined[patch_i:patch_i+CLASSIFICATION_PATCH_SIZE, patch_j:patch_j+CLASSIFICATION_PATCH_SIZE, :] = hypercube_pred
						# patches.append((patch_i, patch_j))
						# print(patch_i, patch_j, hypercube_pred.shape, hypercubeCrop.shape)

						hypercube_combined[u"(%d, %d)"% (patch_i, patch_j)] = hypercube_pred
				print(log_string % (mat_filename+"_patch", end_time, p_mrae/num_patches, p_rrmse/num_patches, p_msam/num_patches, p_sid/num_patches, p_psnr/num_patches, p_ssim/num_patches)) if not mobile_reconstruction else None
				save_mat_patched(os.path.join(OUT_PATH, PATCHED_HS_DIR_NAME, mat_filename), hypercube_combined)
		print(log_string_avg % ("Average %s" % test_dataset, losses_mrae.avg, losses_mrae.stddev, losses_rmse.avg, losses_rmse.stddev,
						  losses_sam.avg, losses_sam.stddev, losses_sid.avg, losses_sid.stddev,
						  losses_psnr.avg, losses_psnr.stddev, losses_ssim.avg, losses_ssim.stddev))
		logger.info(log_string_avg % ("Average %s" % test_dataset, losses_mrae.avg, losses_mrae.stddev, losses_rmse.avg, losses_rmse.stddev,
						  losses_sam.avg, losses_sam.stddev, losses_sid.avg, losses_sid.stddev,
						  losses_psnr.avg, losses_psnr.stddev, losses_ssim.avg, losses_ssim.stddev))
		print("Min Hypercube: %0.9f, Max Hypercube: %0.9f" % (min_hc, max_hc))
		print("Min Predicted Hypercube: %0.9f, Max Predicted Hypercube: %0.9f" % (min_phc, max_phc))
	print(log_string_avg_combined % ("Combined Average %s" % "", losses_mrae_combined.avg, losses_mrae_combined.stddev, losses_rmse_combined.avg, losses_rmse_combined.stddev,
						  losses_sam_combined.avg, losses_sam_combined.stddev, losses_sid_combined.avg, losses_sid_combined.stddev,
						  losses_psnr_combined.avg, losses_psnr_combined.stddev, losses_ssim_combined.avg, losses_ssim_combined.stddev))
	logger.info(log_string_avg_combined % ("Combined Average %s" % "", losses_mrae_combined.avg, losses_mrae_combined.stddev, losses_rmse_combined.avg, losses_rmse_combined.stddev,
						  losses_sam_combined.avg, losses_sam_combined.stddev, losses_sid_combined.avg, losses_sid_combined.stddev,
						  losses_psnr_combined.avg, losses_psnr_combined.stddev, losses_ssim_combined.avg, losses_ssim_combined.stddev))

def main():
	# checkpoint_filename, epoch, iter, model_param, optimizer, val_loss, val_acc = get_best_checkpoint(task="reconstruction")
	# checkpoint_filename = checkpoint_file
	checkpoint_filename = "MSLP_MST++_shelflife_060 trained on all [Half] RGBN (Vanilla).pkl"
	checkpoint = torch.load(os.path.join(MODEL_PATH, "reconstruction", "others", checkpoint_filename))
	model_param = checkpoint["state_dict"]
	model = MST_Plus_Plus(in_channels=4, out_channels=len(BANDS), n_feat=len(BANDS), stage=3)
	model.load_state_dict(model_param)
	model = model.to(device)
	model.eval()
	print(summary(model=model, input_data=(4, 512, 512)))
	inference(model, checkpoint_filename, mobile_reconstruction=False, patched_inference=PATCHED_INFERENCE)

if __name__ == "__main__":
	main()