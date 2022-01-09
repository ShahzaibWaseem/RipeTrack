from __future__ import division

import os
import numpy as np

import torch

from loss import test_mrae, test_rrmse, test_msam, test_sid, test_psnr, test_ssim
from utils import save_matv73, reconstruction, load_mat, initialize_logger
from models.resblock import resblock, conv_bn_relu_res_block

from glob import glob
from imageio import imread

from config import ILLUMINATIONS, TEST_ROOT_DATASET_DIR, TEST_DATASETS, MODEL_PATH, var_name, checkpoint_file, fusion_techniques, init_directories

def main():
	logger = initialize_logger(filename="test.log")
	log_string = "[%s] MRAE=%0.9f, RRMSE=%0.9f, SAM=%0.9f, SID=%0.9f, PSNR=%0.9f, SSIM=%0.9f"

	for fusion in fusion_techniques:
		save_point = torch.load(os.path.join(MODEL_PATH, fusion, checkpoint_file))
		model_param = save_point['state_dict']
		model = resblock(conv_bn_relu_res_block, block_num=10, input_channel=4, output_channel=51, fusion=fusion)
		model.load_state_dict(model_param)
		model = model.cuda()
		model.eval()

		for test_dataset in TEST_DATASETS:
			for illumination in ILLUMINATIONS:
				print("\nFusion: %s\nDataset: %s\nIllumination: %s\nModel: %s\n" % (fusion, test_dataset, illumination, checkpoint_file))
				logger.info("Fusion: %s\tDataset: %s\tIllumination: %s\tModel: %s\n" % (fusion, test_dataset, illumination, checkpoint_file))

				TEST_DATASET_DIR = os.path.join(TEST_ROOT_DATASET_DIR, "working_%s" % test_dataset, "%s_%s_204ch" % (test_dataset, illumination), "test")
				GT_PATH = os.path.join(TEST_DATASET_DIR, "mat")
				IMG_PATH = os.path.join(TEST_DATASET_DIR, "cameraRGBN")
				INF_PATH = os.path.join(TEST_DATASET_DIR, "inference")

				for img_name in glob(os.path.join(IMG_PATH, "*_dense_demRGB.png")):
					if(illumination == "cfl_led"):
						mat_file_name = "_".join(img_name.split("/")[-1].split("_")[0:2])
					else:
						mat_file_name = img_name.split("/")[-1].split("_")[0]

					rgb_img_path = img_name
					nir_img_path = os.path.join(IMG_PATH, img_name.split("/")[-1].replace("RGB", "NIRc"))

					rgb = imread(rgb_img_path)/255

					# flipping red and blue channels (shape used for training)
					rgb[:,:, [0, 2]] = rgb[:,:, [2, 0]]

					nir = imread(nir_img_path)/255

					image = np.dstack((rgb, nir))
					# fixing the dimensions [Channel should be first in torch]
					image = np.expand_dims(np.transpose(image, [2, 1, 0]), axis=0).copy()

					img_res1 = reconstruction(image, model)
					img_res2 = np.flip(reconstruction(np.flip(image, 2).copy(), model), 1)
					inf = (img_res1 + img_res2)/2

					mat_name = "inf_" + mat_file_name + ".mat"
					mat_dir = os.path.join(INF_PATH, fusion, mat_name)
					save_matv73(mat_dir, var_name, inf)

					gt_name = mat_file_name + ".mat"
					gt_dir = os.path.join(GT_PATH, gt_name)
					gt = load_mat(gt_dir, var_name)[var_name][:,:, 1:204:4]

					mrae_error =  test_mrae(inf, gt)
					rrmse_error = test_rrmse(inf, gt)
					sam_error = test_msam(inf, gt)
					sid_error = test_sid(inf, gt)
					psnr_error = test_psnr(inf, gt)
					ssim_error = test_ssim(inf, gt)

					print(log_string % (mat_name, mrae_error, rrmse_error, sam_error, sid_error, psnr_error, ssim_error))
					logger.info(log_string % (mat_name, mrae_error, rrmse_error, sam_error, sid_error, psnr_error, ssim_error))

if __name__ == "__main__":
	init_directories()
	main()