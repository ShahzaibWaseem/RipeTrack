import math
import torch
import numpy as np
from scipy.spatial.distance import cdist

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphNet(nn.Module):
	def __init__(self, img_size=64, input_channel=4, pred_edge=False):
		super(GraphNet, self).__init__()
		self.pred_edge = pred_edge
		self.N = img_size ** 2

		if pred_edge:
			col, row = np.meshgrid(np.arange(img_size), np.arange(img_size))

			coord = np.stack((col, row), axis=2).reshape(-1, 2)
			coord = (coord - np.mean(coord, axis=0)) / (np.std(coord, axis=0) + 1e-5)
			coord = torch.from_numpy(coord).float()  # 4096, 2
			coord = torch.cat((coord.unsqueeze(0).repeat(self.N, 1,  1),
									coord.unsqueeze(1).repeat(1, self.N, 1)), dim=2)
			# coord = torch.abs(coord[:, :, [0, 1]] - coord[:, :, [2, 3]])
			self.pred_edge_fc = nn.Sequential(nn.Linear(4, 16),
											  nn.ReLU(),
											  nn.Linear(16, 1),
											  nn.Tanh())
			# self.pred_edge_fc = nn.Sequential(nn.Conv2d(4, 64, 3))
			self.register_buffer('coord', coord)
		else:
			# precompute adjacency matrix before training
			A = self.precompute_adjacency_images(img_size)
			self.register_buffer('A', A)

	@staticmethod
	def precompute_adjacency_images(img_size):
		col, row = np.meshgrid(np.arange(img_size), np.arange(img_size))
		coord = np.stack((col, row), axis=2).reshape(-1, 2) / img_size
		dist = cdist(coord, coord)  
		sigma = 0.05 * np.pi

		A = np.exp(- dist / sigma ** 2)
		# print('WARNING: try squaring the dist to make it a Gaussian')
			
		A[A < 0.01] = 0
		A = torch.from_numpy(A).float()

		# Normalization as per (Kipf & Welling, ICLR 2017)
		D = A.sum(1)  # nodes degree (N,)
		D_hat = (D + 1e-5) ** (-0.5)
		A_hat = D_hat.view(-1, 1) * A * D_hat.view(1, -1)  # N,N

		# Some additional trick I found to be useful
		A_hat[A_hat > 0.0001] = A_hat[A_hat > 0.0001] - 0.2

		return A_hat

	def forward(self, x):
		# print("graph shape", x.shape)
		B = x.size(0)
		img_size = x.size(2)
		C = x.size(1)

		if self.pred_edge:
			self.A = self.pred_edge_fc(self.coord).squeeze()

		gnn_out = torch.cuda.FloatTensor([B, C, img_size, img_size])
		adj=self.A.unsqueeze(0).expand(B, -1, -1)
		inp=x.view(B, -1, C)
		# print(adj.shape, inp.shape)

		for channel in range(C):
			inp_c = inp[:,:,channel].unsqueeze(2)
			# print("adj, inp", adj.shape, inp_c.shape)

			avg_neighbor_features = torch.bmm(adj, inp_c)
			# print("bmm", avg_neighbor_features.shape)
			
			avg_neighbor_features = avg_neighbor_features.view(B, img_size, img_size)
			# print("view", avg_neighbor_features.shape)

			avg_neighbor_features = avg_neighbor_features.unsqueeze(1)
			# print("unsq", avg_neighbor_features.shape)
			if channel == 0:
				gnn_out = avg_neighbor_features
			else:
				gnn_out = torch.cat((gnn_out, avg_neighbor_features), 1)

		#cat
		# print(gnn_out.shape)
		# avg_neighbor_features = (torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1),
		# 						 x.view(B, -1, 1)).view(B, -1))

		# avg_neighbor_features = torch.bmm(self.A, x)
		return gnn_out