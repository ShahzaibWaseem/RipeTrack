from __future__ import division

import os
import time
import numpy as np

import torch
import torch.nn as nn

from loss import mrae, rmse
from utils import save_matv73, reconstruction, load_mat
from models.resblock import resblock, conv_bn_relu_res_block

from glob import glob
from imageio import imread

from config import GT_PATH, INF_PATH, IMG_PATH, MODEL_PATH, var_name, checkpoint_file

save_point = torch.load(os.path.join(MODEL_PATH, checkpoint_file))
model_param = save_point['state_dict']
model = resblock(conv_bn_relu_res_block, 10, 4, 51)
model.load_state_dict(model_param)

model = model.cuda()
model.eval()

for img_name in glob(os.path.join(IMG_PATH, "*_dense_demRGB.png")):
	mat_file_name = img_name.split("/")[-1].split("_")[0]
	rgb_img_path = img_name
	nir_img_path = os.path.join(IMG_PATH, img_name.split("/")[-1].replace("RGB", "NIRc"))

	rgb = imread(rgb_img_path)
	# rgb = np.resize(rgb, (64, 64, 3))
	rgb = rgb/255

	nir = imread(nir_img_path)
	# nir = np.resize(nir, (64, 64, 1))
	nir = nir/255

	image = np.dstack((rgb, nir))

	image = np.expand_dims(np.transpose(image,[2,1,0]), axis=0).copy()

	img_res1 = reconstruction(image,model)
	img_res2 = np.flip(reconstruction(np.flip(image, 2).copy(),model),1)
	img_res3 = (img_res1+img_res2)/2

	mat_name = "inf_" + mat_file_name + '.mat'
	mat_dir= os.path.join(INF_PATH, mat_name)

	save_matv73(mat_dir, var_name,img_res3)

	gt_name = mat_file_name + '.mat'
	gt_dir= os.path.join(GT_PATH, gt_name)
	gt = load_mat(gt_dir,var_name)
	mrae_error =  mrae(img_res3, gt[var_name][:,:,1:204:4])
	rrmse_error = rmse(img_res3, gt[var_name][:,:,1:204:4])

	print("[%s] MRAE=%0.9f RRMSE=%0.9f" %(img_name,mrae_error,rrmse_error))