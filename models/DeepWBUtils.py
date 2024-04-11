"""
	Deep white-balance editing main function (inference phase)
	Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
	If you use this code, please cite the following paper:
	Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

def deep_wb(image, task='all', net_awb=None, net_t=None, net_s=None, device='cpu', s=656):
	# check image size
	image_resized = image.resize((round(image.width / max(image.size) * s), round(image.height / max(image.size) * s)))
	w, h = image_resized.size
	if w % 2 ** 4 == 0:
		new_size_w = w
	else:
		new_size_w = w + 2 ** 4 - w % 2 ** 4

	if h % 2 ** 4 == 0:
		new_size_h = h
	else:
		new_size_h = h + 2 ** 4 - h % 2 ** 4

	inSz = (new_size_w, new_size_h)
	if not ((w, h) == inSz):
		image_resized = image_resized.resize(inSz)

	image = np.array(image)
	image_resized = np.array(image_resized)
	img = image_resized.transpose((2, 0, 1))
	img = img / 255
	img = torch.from_numpy(img)
	img = img.unsqueeze(0)
	img = img.to(device=device, dtype=torch.float32)
	tf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

	if task == 'all':
		net_awb.eval()
		net_t.eval()
		net_s.eval()
		with torch.no_grad():
			output_awb = net_awb(img)
			output_t = net_t(img)
			output_s = net_s(img)

		output_awb = tf(torch.squeeze(output_awb.cpu()))
		output_awb = output_awb.squeeze().cpu().numpy()
		output_awb = output_awb.transpose((1, 2, 0))
		m_awb = get_mapping_func(image_resized, output_awb)
		output_awb = outOfGamutClipping(apply_mapping_func(image, m_awb))

		output_t = tf(torch.squeeze(output_t.cpu()))
		output_t = output_t.squeeze().cpu().numpy()
		output_t = output_t.transpose((1, 2, 0))
		m_t = get_mapping_func(image_resized, output_t)
		output_t = outOfGamutClipping(apply_mapping_func(image, m_t))

		output_s = tf(torch.squeeze(output_s.cpu()))
		output_s = output_s.squeeze().cpu().numpy()
		output_s = output_s.transpose((1, 2, 0))
		m_s = get_mapping_func(image_resized, output_s)
		output_s = outOfGamutClipping(apply_mapping_func(image, m_s))

		return output_awb, output_t, output_s

def outOfGamutClipping(I):
	""" Clips out-of-gamut pixels. """
	I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
	I[I < 0] = 0  # any pixel is below 0, clip it to 0
	return I

def kernelP(I):
	""" Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
		Ref: Hong, et al., "A study of digital camera colorimetric characterization
		based on polynomial modeling." Color Research & Application, 2001. """
	return (np.transpose((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
						  I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
						  I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
						  np.repeat(1, np.shape(I)[0]))))

def get_mapping_func(image1, image2):
	""" Computes the polynomial mapping """
	image1 = np.reshape(image1, [-1, 3])
	image2 = np.reshape(image2, [-1, 3])
	m = LinearRegression().fit(kernelP(image1), image2)
	return m

def apply_mapping_func(image, m):
	""" Applies the polynomial mapping """
	sz = image.shape
	image = np.reshape(image, [-1, 3])
	result = m.predict(kernelP(image))
	result = np.reshape(result, [sz[0], sz[1], sz[2]])
	return result

def colorTempInterpolate(I_T, I_S):
	""" Interpolates between tungsten and shade WB to produce Cloudy, Daylight, and Fluorescent WB """
	ColorTemperatures = {'T': 2850, 'F': 3800, 'D': 5500, 'C': 6500, 'S': 7500}
	cct1 = ColorTemperatures['T']
	cct2 = ColorTemperatures['S']
	# interpolation weight
	cct1inv = 1 / cct1
	cct2inv = 1 / cct2
	tempinv_F = 1 / ColorTemperatures['F']
	tempinv_D = 1 / ColorTemperatures['D']
	tempinv_C = 1 / ColorTemperatures['C']
	g_F = (tempinv_F - cct2inv) / (cct1inv - cct2inv)
	g_D = (tempinv_D - cct2inv) / (cct1inv - cct2inv)
	g_C = (tempinv_C - cct2inv) / (cct1inv - cct2inv)
	I_F = g_F * I_T + (1 - g_F) * I_S
	I_D = g_D * I_T + (1 - g_D) * I_S
	I_C = g_C * I_T + (1 - g_C) * I_S
	return I_F, I_D, I_C

def colorTempInterpolate_w_target(I_T, I_S, target_temp):
	""" Interpolates between tungsten and shade WB to produce target_temp WB """
	cct1 = 2850
	cct2 = 7500
	# interpolation weight
	cct1inv = 1 / cct1
	cct2inv = 1 / cct2
	tempinv_target = 1 / target_temp
	g = (tempinv_target - cct2inv) / (cct1inv - cct2inv)
	return g * I_T + (1 - g) * I_S

def to_image(image):
	""" converts to PIL image """
	return Image.fromarray((image * 255).astype(np.uint8))

def imshow(img, *arguments, colortemp=None):
	""" displays image """
	outimgs_num = 0
	for _ in arguments:
		outimgs_num += 1

	if outimgs_num == 1 and not colortemp:
		titles = ["input", "awb"]
	elif outimgs_num == 1 and colortemp:
		titles = ["input", "output (%dK)" % colortemp]
	elif outimgs_num == 5:
		titles = ["input", "tungsten", "fluorescent", "daylight", "cloudy", "shade"]
	elif outimgs_num == 6:
		titles = ["input", "awb", "tungsten", "fluorescent", "daylight", "cloudy", "shade"]
	else:
		raise Exception('Unexpected number of output images')

	if outimgs_num < 3:
		fig, ax = plt.subplots(1, outimgs_num + 1)
		ax[0].set_title(titles[0])
		ax[0].imshow(img)
		ax[0].axis('off')
		i = 1
		for image in arguments:
			if outimgs_num < 3:
				ax[i].set_title(titles[i])
				ax[i].imshow(image)
				ax[i].axis('off')
				i = i + 1
	else:
		fig, ax = plt.subplots(2 + (outimgs_num + 1) % 3, 3)
		ax[0][0].set_title(titles[0])
		ax[0][0].imshow(img)
		ax[0][0].axis('off')
		i = 1
		for image in arguments:
			if i == outimgs_num and outimgs_num == 6:
				ax[2][1].set_title(titles[i])
				ax[2][1].imshow(image)
				ax[2][1].axis('off')
				ax[2][0].axis('off')
			else:
				ax[i // 3][i % 3].set_title(titles[i])
				ax[i // 3][i % 3].imshow(image)
				ax[i // 3][i % 3].axis('off')
			i = i + 1

	plt.xticks([]), plt.yticks([])
	plt.axis('off')
	plt.show()