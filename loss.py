#!/usr/local/bin/python
from __future__ import division

import numpy as np
from skimage.metrics import structural_similarity

import torch
import torch.nn as nn
from torch.autograd import Variable

class reconstruct_loss(nn.Module):
	"""the loss between the input and synthesized input"""
	def __init__(self, cie_matrix, batchsize):
		super(reconstruct_loss, self).__init__()
		self.cie = Variable(torch.from_numpy(cie_matrix).float().cuda(), requires_grad=False)
		self.batchsize = batchsize
	def forward(self, network_input, network_output):
		network_output = network_output.permute(3, 2, 0, 1)
		network_output = network_output.contiguous().view(-1, 31)
		reconsturct_input = torch.mm(network_output,self.cie)
		reconsturct_input = reconsturct_input.view(50, 50, 64, 3)
		reconsturct_input = reconsturct_input.permute(2,3,1,0)
		reconstruction_loss = torch.mean(torch.abs(reconsturct_input - network_input))
		return reconstruction_loss

def mrae_loss(tensor_pred, tensor_gt):
	""" Computes the MRAE value (Training Loss) """
	error = torch.abs(tensor_pred-tensor_gt)/tensor_gt
	rrmse = torch.mean(error.view(-1))
	return rrmse

def sam_loss(tensor_pred, tensor_gt):
	""" Spectral Angle Mapper (Training Loss) """
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

def test_mrae(img_pred, img_gt):
	"""Calculate the relative MRAE"""
	error = img_pred - img_gt
	error_relative = error/img_gt
	rrmse = np.mean(np.abs(error_relative))
	return rrmse

def test_rmse(img_pred, img_gt):
	error = img_pred - img_gt
	error_relative = error/img_gt
	rrmse = np.sqrt(np.mean((np.power(error_relative, 2))))
	return rrmse

def spectral_angle(a, b):
	""" Spectral angle """
	va = a / np.sqrt(a.dot(a))
	vb = b / np.sqrt(b.dot(b))
	# print(va.shape, vb.shape)
	return np.arccos(va.dot(vb))

def test_msam(img_pred, img_gt):
	""" mean spectral angle mapper """
	img_pred_flat = img_pred.reshape(-1, img_pred.shape[2])
	img_gt_flat = img_gt.reshape(-1, img_gt.shape[2])
	assert len(img_pred_flat) == len(img_gt_flat)
	return np.mean([spectral_angle(img_pred_flat[i]/4095, img_gt_flat[i]/4095) for i in range(len(img_pred_flat))])

def spectral_divergence(a, b):
	""" Spectral Divergence """
	p = (a / np.sum(a)) + np.spacing(1)
	q = (b / np.sum(b)) + np.spacing(1)
	return np.sum(p * np.log(p / q) + q * np.log(q / p))

def test_sid(img_pred, img_gt):
	""" mean spectral information divergence """
	img_pred_flat = img_pred.reshape(-1, img_pred.shape[2])
	img_gt_flat = img_gt.reshape(-1, img_gt.shape[2])
	assert len(img_pred_flat) == len(img_gt_flat)
	return np.mean([spectral_divergence(img_pred_flat[i]/4095, img_gt_flat[i]/4095) for i in range(len(img_pred_flat))])

def mse(img_pred, img_gt):
	error = (img_pred - img_gt)
	mse = np.mean((np.power(error, 2)))
	return mse

def test_psnr(img_pred, img_gt):
	return 10 * np.log10(4095**2 / mse(img_pred, img_gt))

def test_ssim(img_pred, img_gt, max_p=4095):
	"""
	Structural Simularity Index
	"""
	return structural_similarity(img_gt, img_pred, data_range=max_p, channel_axis=True)