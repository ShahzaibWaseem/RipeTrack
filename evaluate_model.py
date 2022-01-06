from __future__ import division

import os
import numpy as np

import torch

from loss import test_mrae, test_rmse, test_msam
from utils import save_matv73, reconstruction, load_mat
from models.resblock import resblock, conv_bn_relu_res_block

from glob import glob
from imageio import imread

from config import ILLUMINATIONS, TEST_ROOT_DATASET_DIR, TEST_DATASETS, MODEL_PATH, var_name, checkpoint_file, fusion_techniques, init_directories

def main():
	for fusion in fusion_techniques:
		save_point = torch.load(os.path.join(MODEL_PATH, fusion, checkpoint_file))
		model_param = save_point['state_dict']

		model = resblock(conv_bn_relu_res_block, block_num=10, input_channel=4, output_channel=51, fusion=fusion)
		model.load_state_dict(model_param)

		model = model.cuda()
		model.eval()

		for test_dataset in TEST_DATASETS:
			for illumination in ILLUMINATIONS:
				print("\nFusion: %s\nDataset: %s\nIllumination: %s\n" % (fusion, test_dataset, illumination))
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

					rgb = imread(rgb_img_path)
					
					rgb_new = np.zeros(rgb.shape)
					rgb_new[:,:,0] = rgb[:,:,2]
					rgb_new[:,:,1] = rgb[:,:,1]
					rgb_new[:,:,2] = rgb[:,:,0]
					rgb = rgb_new

					rgb = rgb/255

					nir = imread(nir_img_path)/255

					image = np.dstack((rgb, nir))
					image = np.expand_dims(np.transpose(image,[2,1,0]), axis=0).copy()

					img_res1 = reconstruction(image,model)
					img_res2 = np.flip(reconstruction(np.flip(image, 2).copy(),model),1)
					img_res3 = (img_res1+img_res2)/2

					mat_name = "inf_" + mat_file_name + '.mat'
					mat_dir = os.path.join(INF_PATH, fusion, mat_name)
					save_matv73(mat_dir, var_name, img_res3)

					gt_name = mat_file_name + '.mat'
					gt_dir = os.path.join(GT_PATH, gt_name)
					gt = load_mat(gt_dir, var_name)
					mrae_error =  test_mrae(img_res3, gt[var_name][:,:,1:204:4])
					rrmse_error = test_rmse(img_res3, gt[var_name][:,:,1:204:4])
					sam_error = test_msam(img_res3, gt[var_name][:,:,1:204:4])

					print("[%s] MRAE=%0.9f, RRMSE=%0.9f, SAM=%0.9f" %(img_name, mrae_error, rrmse_error, sam_error))

if __name__ == "__main__":
	init_directories()
	main()