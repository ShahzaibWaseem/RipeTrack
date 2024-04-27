#!/usr/local/bin/python
from __future__ import division

import numpy as np
from skimage.metrics import structural_similarity

import torch
import torch.nn as nn

class Loss_MRAE(nn.Module):
	""" Computes the Mean Relative Absolute Error Loss (PyTorch - Training Loss) """
	def __init__(self):
		super(Loss_MRAE, self).__init__()

	def forward(self, tensor_pred, tensor_gt):
		assert tensor_pred.shape == tensor_gt.shape
		error = torch.abs((tensor_pred-tensor_gt)/tensor_gt)
		mrae = torch.mean(error.reshape(-1))
		return mrae

class Loss_SAM(nn.Module):
	""" Computes the Spectral Angle Mapper Loss (PyTorch - Training Loss) """
	def __init__(self):
		super(Loss_SAM, self).__init__()

	def forward(self, tensor_pred, tensor_gt):
		assert tensor_pred.shape == tensor_gt.shape
		EPS = 1e-7
		# inner product
		dot = torch.sum(tensor_pred * tensor_gt, dim=1).view(-1)
		# norm calculations
		image = tensor_pred.reshape(-1, tensor_pred.shape[1])
		norm_original = torch.norm(image, p=2, dim=1)

		target = tensor_gt.reshape(-1, tensor_gt.shape[1])
		norm_reconstructed = torch.norm(target, p=2, dim=1)

		norm_product = (norm_original.mul(norm_reconstructed)).pow(-1)
		argument = dot.mul(norm_product)
		# for avoiding arccos(1)
		acos = torch.acos(torch.clamp(argument, min=-1+EPS, max=1-EPS))
		loss = torch.mean(acos)

		if torch.isnan(loss):
			raise ValueError(f"Loss is NaN value. Consecutive values - dot: {dot},\
				norm original: {norm_original}, norm reconstructed: {norm_reconstructed},\
				norm product: {norm_product}, argument: {argument}, acos: {acos},\
				loss: {loss}, input: {tensor_pred}, output: {target}")
		return loss

class Loss_SID(nn.Module):
	""" Computes the Spectral Information Divergence Loss (PyTorch - Training Loss) """
	def __init__(self):
		super(Loss_SID, self).__init__()

	def forward(self, tensor_pred, tensor_gt):
		assert tensor_pred.shape == tensor_gt.shape
		EPS = 1e-3
		output = torch.clamp(tensor_pred, 0, 1)
		a1 = output * torch.log10((output + EPS) / (tensor_gt + EPS))
		a2 = tensor_gt * torch.log10((tensor_gt + EPS) / (output + EPS))

		a1_sum = a1.sum(dim=3).sum(dim=2)
		a2_sum = a2.sum(dim=3).sum(dim=2)

		sid = torch.mean(torch.abs(a1_sum + a2_sum))
		if torch.isnan(sid):
			raise ValueError(f"Loss is NaN value. output: {output},\
				a1: {a1}, a1_sum: {a1_sum},\
				a2: {a2}, a2_sum: {a2_sum},\
				sid: {sid}, input: {tensor_pred}, output: {tensor_gt}")
		return sid

def mse(img_pred, img_gt):
	""" Calculate the mean square error (NumPy - used in test_psnr())"""
	error = img_pred - img_gt
	mse = np.mean(np.power(error, 2))
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

def test_mrae(img_pred, img_gt, relative=True):
	""" Calculate the relative Mean Relative Absolute Error (NumPy - Test Error) """
	error = img_pred - img_gt
	error_relative = error/img_gt if relative else error
	mrae = np.mean(np.abs(error_relative))
	return mrae

def test_rrmse(img_pred, img_gt, relative=False):
	""" Calculate the relative Root Mean Square Error (NumPy - Test Error) """
	error = img_pred - img_gt
	error_relative = error/img_gt if relative else error
	rrmse = np.sqrt(np.mean(np.power(error_relative, 2)))
	return rrmse

def test_msam(img_pred, img_gt, max_value=1.0):
	""" Calculate the mean spectral angle mapper (NumPy - Test Error) """
	img_pred_flat = img_pred.reshape(-1, img_pred.shape[2])
	img_gt_flat = img_gt.reshape(-1, img_gt.shape[2])
	assert len(img_pred_flat) == len(img_gt_flat)
	return np.mean([spectral_angle(img_pred_flat[i]/max_value, img_gt_flat[i]/max_value) for i in range(len(img_pred_flat))])

def test_sid(img_pred, img_gt, max_value=1.0):
	""" mean spectral information divergence """
	img_pred_flat = img_pred.reshape(-1, img_pred.shape[2])
	img_gt_flat = img_gt.reshape(-1, img_gt.shape[2])
	assert len(img_pred_flat) == len(img_gt_flat)
	return np.mean([spectral_divergence(img_pred_flat[i]/max_value, img_gt_flat[i]/max_value) for i in range(len(img_pred_flat))])

def test_psnr(img_pred, img_gt, max_value=1.0):
	""" Calculate the peak signal to noise ratio (NumPy - Test Error) """
	return 10 * np.log10(max_value**2 / mse(img_pred, img_gt))

def test_ssim(img_pred, img_gt, max_value=1.0):
	""" Calculate the structural simularity index measure (NumPy - Test Error) """
	return structural_similarity(img_gt, img_pred, data_range=max_value, channel_axis=True)

def test_ssim_db(img_pred, img_gt, max_value=1.0):
	""" Calculate the structural simularity index measure in decibels (NumPy - Test Error) """
	ssim = test_ssim(img_pred, img_gt, max_value)
	return -10 * np.log10(1- ssim)