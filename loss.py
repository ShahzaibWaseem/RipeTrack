#!/usr/local/bin/python
from __future__ import division

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
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

def rrmse_loss(outputs, label):
	"""Computes the rrmse value"""
	error = torch.abs(outputs-label)/label
	rrmse = torch.mean(error.view(-1))
	return rrmse

def mrae(img_res,img_gt):
	"""Calculate the relative RMSE"""
	error= img_res- img_gt
	error_relative = error/img_gt
	rrmse = np.mean((np.sqrt(np.power(error_relative, 2))))
	return rrmse

def rmse(img_res,img_gt):
	error= img_res- img_gt
	error_relative = error/img_gt
	rrmse =np.sqrt(np.mean((np.power(error_relative, 2))))
	return rrmse

def sam_loss(input_tensor, target_tensor):
	# inner product
	dot = torch.sum(input_tensor * target_tensor, dim=1).view(-1)
	# norm calculations
	image = input_tensor.view(-1, input_tensor.shape[1])
	norm_original = torch.norm(image, p=2, dim=1)

	target = target_tensor.view(-1, target_tensor.shape[1])
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
			loss: {loss}, input: {input_tensor}, output: {target}"
		)
	return loss

def spectral_angle(a, b):
	""" Spectral angle """
	va = a / np.sqrt(a.dot(a))
	vb = b / np.sqrt(b.dot(b))
	return np.arccos(np.clip(va.dot(vb), -1, 1))

def val_msad(X, Y):
	""" mean spectral angle """
	assert len(X) == len(Y)
	return np.mean([spectral_angle(X[i], Y[i]) for i in range(len(X))])