""" Depreciated: file will be removed in the future """
import os
import sys
sys.path.append(os.path.join(".."))

import numpy as np
from imageio import imread

import torch

from models.MST import MST_Plus_Plus

from config import MODEL_PATH, BANDS, checkpoint_file

import matplotlib.pyplot as plt

activation = {}

def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.cpu().detach()
	return hook
	
def visualize_features():
	IMG_PATH = os.path.join("..", "data_preparation", "working_datasets", "working_avocado", "avocado_cfl_led_204ch", "test", "cameraRGBN")

	save_point = torch.load(os.path.join(MODEL_PATH, checkpoint_file))
	model_param = save_point["state_dict"]
	model = MST_Plus_Plus(in_channels=4, out_channels=len(BANDS), n_feat=len(BANDS), stage=3)
	model.load_state_dict(model_param)
	model = model.cuda()
	model.eval()
	# print(summary(model, (4, 512, 512)))

	model.conv_seq[0].register_forward_hook(get_activation("conv_seq"))

	rgb = imread(os.path.join(IMG_PATH, "1461_f_dense_demRGB.png"))/255
	rgb[:,:, [0, 2]] = rgb[:,:, [2, 0]]		# flipping red and blue channels (shape used for training)

	nir_img_path = os.path.join(IMG_PATH, "1461_f_dense_demNIRc.png")
	nir = imread(nir_img_path)/255

	image = np.dstack((rgb, nir))
	image = np.expand_dims(np.transpose(image, [2, 1, 0]), axis=0).copy()	# fixing the dimensions [Channel should be first in torch]

	output = model(torch.autograd.Variable(torch.from_numpy(image).float()).cuda())
	print(activation["conv_seq"].shape)
	
	act = activation["conv_seq"].squeeze()
	print(act.shape)

	fig, axs = plt.subplots(act.size(0)//8, act.size(0)//8)
	iter = 0
	for i in range(axs.shape[0]):
		for j in range(axs.shape[1]):
			axs[i, j].imshow(act[iter], cmap="gray")
			iter = iter + 1
	plt.show()

if __name__ == "__main__":
	os.chdir("..")
	visualize_features()