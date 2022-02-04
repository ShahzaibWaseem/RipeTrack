#!/usr/local/bin/python
from __future__ import division

import numpy as np
from skimage.metrics import structural_similarity

import torch

def mrae_loss(tensor_pred, tensor_gt):
	""" Computes the Mean Relative Absolute Error Loss (PyTorch - Training Loss) """
	error = torch.abs(tensor_pred-tensor_gt)/tensor_gt
	rrmse = torch.mean(error.view(-1))
	return rrmse

def sam_loss(tensor_pred, tensor_gt):
	""" Computes the Spectral Angle Mapper Loss (PyTorch - Training Loss) """
	# inner product
	dot = torch.sum(tensor_pred * tensor_gt, dim=1).view(-1)
	# norm calculations
	image = tensor_pred.view(-1, tensor_pred.shape[1])
	norm_original = torch.norm(image, p=2, dim=1)

	target = tensor_gt.view(-1, tensor_gt.shape[1])
	norm_reconstructed = torch.norm(target, p=2, dim=1)

	norm_product = (norm_original.mul(norm_reconstructed)).pow(-1)
	argument = dot.mul(norm_product)
	# for avoiding arccos(1)
	acos = torch.acos(torch.clamp(argument, -1 + 1e-7, 1 - 1e-7))
	loss = torch.mean(acos)

	if torch.isnan(loss):
		raise ValueError(
			f"Loss is NaN value. Consecutive values - dot: {dot}, \
			norm original: {norm_original}, norm reconstructed: {norm_reconstructed}, \
			norm product: {norm_product}, argument: {argument}, acos: {acos}, \
			loss: {loss}, input: {tensor_pred}, output: {target}"
		)
	return loss

def mse(img_pred, img_gt):
	""" Calculate the mean square error (NumPy - used in test_psnr())"""
	error = (img_pred - img_gt)
	mse = np.mean((np.power(error, 2)))
	return mse

def spectral_angle(a, b):
	""" Spectral angle between two arrays (NumPy - used in test_msam()) """
	va = a / np.sqrt(a.dot(a))
	vb = b / np.sqrt(b.dot(b))
	return np.arccos(va.dot(vb))

def spectral_divergence(a, b):
	""" Spectral Divergence between two arrays (NumPy - used in test_sid()) """
	p = (a / np.sum(a)) + np.spacing(1)
	q = (b / np.sum(b)) + np.spacing(1)
	return np.sum(p * np.log(p / q) + q * np.log(q / p))

def test_mrae(img_pred, img_gt):
	""" Calculate the relative Mean Relative Absolute Error (NumPy - Test Error) """
	error = img_pred - img_gt
	error_relative = error/img_gt
	rrmse = np.mean(np.abs(error_relative))
	return rrmse

def test_rrmse(img_pred, img_gt):
	""" Calculate the relative Root Mean Square Error (NumPy - Test Error) """
	error = img_pred - img_gt
	error_relative = error/img_gt
	rrmse = np.sqrt(np.mean((np.power(error_relative, 2))))
	return rrmse

def test_msam(img_pred, img_gt):
	""" Calculate the mean spectral angle mapper (NumPy - Test Error) """
	img_pred_flat = img_pred.reshape(-1, img_pred.shape[2])
	img_gt_flat = img_gt.reshape(-1, img_gt.shape[2])
	assert len(img_pred_flat) == len(img_gt_flat)
	return np.mean([spectral_angle(img_pred_flat[i]/4095, img_gt_flat[i]/4095) for i in range(len(img_pred_flat))])

def test_sid(img_pred, img_gt):
	""" mean spectral information divergence """
	img_pred_flat = img_pred.reshape(-1, img_pred.shape[2])
	img_gt_flat = img_gt.reshape(-1, img_gt.shape[2])
	assert len(img_pred_flat) == len(img_gt_flat)
	return np.mean([spectral_divergence(img_pred_flat[i]/4095, img_gt_flat[i]/4095) for i in range(len(img_pred_flat))])

def test_psnr(img_pred, img_gt):
	""" Calculate the peak signal to noise ratio (NumPy - Test Error) """
	return 10 * np.log10(4095**2 / mse(img_pred, img_gt))

def test_ssim(img_pred, img_gt, max_p=4095):
	""" Calculate the structural simularity index measure (NumPy - Test Error) """
	return structural_similarity(img_gt, img_pred, data_range=max_p, channel_axis=True)