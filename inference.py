from __future__ import division

import os
import time
import numpy as np
from glob import glob

import torch
from torchsummary import summary

from models.model import Network
from models.resblock import ResNeXtBottleneck

from train import get_required_transforms
from dataset import read_image
from utils import save_matv73, reconstruction, load_mat, initialize_logger
from loss import test_mrae, test_rrmse, test_msam, test_sid, test_psnr, test_ssim
from config import ILLUMINATIONS, TEST_ROOT_DATASET_DIR, TEST_DATASETS, MODEL_PATH, MODEL_NAME, APPLICATION_NAME, BAND_SPACING, RGBN_BANDS, NUMBER_OF_BANDS, BANDS, model_run_title, checkpoint_file, create_directory, init_directories

import matplotlib.pyplot as plt

def calculate_metrics(img_pred, img_gt):
	mrae = test_mrae(img_pred, img_gt)
	rrmse = test_rrmse(img_pred, img_gt)
	msam = test_msam(img_pred, img_gt)
	sid = test_sid(img_pred, img_gt)
	psnr = test_psnr(img_pred, img_gt)
	ssim = test_ssim(img_pred, img_gt)
	return mrae, rrmse, msam, sid, psnr, ssim

def inference(model, rgbn_from_cube=True):
	input_transform, label_transform = get_required_transforms(task="reconstruction")
	eps = 1e-5
	logger = initialize_logger(filename="test.log")
	log_string = "[%15s] Time: %0.9f, MRAE: %0.9f, RRMSE: %0.9f, SAM: %0.9f, SID: %0.9f, PSNR: %0.9f, SSIM: %0.9f"
	TEST_DATASET_DIR = os.path.join(TEST_ROOT_DATASET_DIR, "working_%s" % APPLICATION_NAME)

	for test_dataset in TEST_DATASETS:
		for illumination in ILLUMINATIONS:
			print("\n" + model_run_title)
			logger.info(model_run_title)
			print("\nDataset: %s\nIllumination: %s\nModel: %s\n" % (APPLICATION_NAME, illumination, checkpoint_file))
			logger.info("Dataset: %s\tIllumination: %s\tModel: %s\n" % (APPLICATION_NAME, illumination, checkpoint_file))

			GT_PATH = os.path.join(TEST_DATASET_DIR, "working_%s_204ch" % test_dataset)
			IMG_PATH = os.path.join(os.path.dirname(TEST_DATASET_DIR), "RGBNIRImages", "working_%s" % APPLICATION_NAME, "working_%s_204ch" % test_dataset) if not rgbn_from_cube else None
			INF_PATH = os.path.join(GT_PATH, "inference")

			for mat_filepath in sorted(glob(os.path.join(TEST_DATASET_DIR, "working_%s_204ch" % test_dataset, "*.mat"))):
				start_time = time.time()
				label = mat_filepath.split("/")[-2]

				hypercube = load_mat(mat_filepath)
				mat_filename = "_".join(mat_filepath.split("/")[-1].split("_")[0:2]) if illumination == "cfl_led" \
							   else mat_filepath.split("/")[-1].split("_")[0]

				if not rgbn_from_cube:
					rgb_filename = mat_filename.split(".")[0] + "_RGB.png"
					nir_filename = mat_filename.split(".")[0] + "_NIR.png"
					rgb_filepath = os.path.join(IMG_PATH, rgb_filename)
					nir_filepath = os.path.join(IMG_PATH, nir_filename)
					image = read_image(rgb_filepath, nir_filepath)
					image = np.transpose(image, [2, 0, 1])
				else:
					image = hypercube[:, :, RGBN_BANDS]
					image = np.transpose(image, [2, 0, 1])

				# Image Min: -1.1055755615234375, Hypercube Min: -1.3652015924453735, Inference Min: -1.7359950542449951
				image = torch.from_numpy(np.expand_dims(image, axis=0)).float()
				image = input_transform(image).cuda()
				image = image + 1.1055755615234375 + 0.1

				hypercube = hypercube[:, :, BANDS]
				hypercube = np.transpose(hypercube, [2, 0, 1])
				hypercube = torch.from_numpy(hypercube).float()
				hypercube = label_transform(hypercube)
				hypercube = np.transpose(hypercube.numpy(), [1, 2, 0])
				hypercube = hypercube + 1.3652015924453735 + 0.1

				# hypercube_pred = (reconstruction(image, model) + np.flip(reconstruction(np.flip(image, 2).copy(), model), 1))
				hypercube_pred = model(image)
				hypercube_pred = np.transpose(hypercube_pred.squeeze(0).cpu().detach().numpy(), [1, 2, 0])

				# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
				# ax1.imshow(np.transpose(image.detach().cpu().numpy().squeeze(0), [1, 2, 0])[:, :, :3])
				# ax1.set_title("Input")
				# ax2.imshow(hypercube[:, :, 50])
				# ax2.set_title("Ground Truth")
				# ax3.imshow(hypercube_pred[:, :, 50])
				# ax3.set_title("Output")
				# plt.show()

				end_time = time.time() - start_time

				create_directory(INF_PATH)

				inf_mat_name = os.path.join(TEST_DATASET_DIR, label, "inference", "inf_" + mat_filename)
				save_matv73(inf_mat_name, hypercube_pred)

				mrae, rrmse, msam, sid, psnr, ssim = calculate_metrics(hypercube_pred+eps, hypercube+eps)

				print(log_string % (mat_filename, end_time, mrae, rrmse, msam, sid, psnr, ssim))
				logger.info(log_string % (mat_filename, end_time, mrae, rrmse, msam, sid, psnr, ssim))

def main():
	save_point = torch.load(os.path.join(MODEL_PATH, checkpoint_file))
	model_param = save_point["state_dict"]
	model = Network(ResNeXtBottleneck, block_num=10, input_channel=4, n_hidden=64, output_channel=len(BANDS))
	model.load_state_dict(model_param)
	model = model.cuda()
	model.eval()
	print(summary(model=model, input_data=(4, 512, 512)))
	inference(model, rgbn_from_cube=False)

if __name__ == "__main__":
	# init_directories()
	main()