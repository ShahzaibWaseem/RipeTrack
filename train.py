from __future__ import division

import  os
import time

from skimage.segmentation import slic

import torch
import torch.nn as nn

from torchvision.datasets import MNIST
from torchsummary import summary
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset

from loss import mrae_loss, sam_loss
from dataset import DatasetFromHdf5, DatasetFromDirectory
from models.resblock import resblock, ResNeXtBottleneck
from models.model import Network

from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, make_h5_dataset, modeltoONNX, ONNXtotf, tf_to_tflite

from config import ILLUMINATIONS, TRAIN_DATASET_DIR, TEST_ROOT_DATASET_DIR, TRAIN_DATASET_FILES, VALID_DATASET_FILES, TEST_DATASETS, LOGS_PATH, MODEL_NAME, DATASET_NAME, init_directories, fusion_techniques, batch_size, end_epoch, init_lr, model_run_title

def main():
	torch.backends.cudnn.benchmark = True

	# Dataset
	train_data, valid_data = [], []

	# for datasetFile in TRAIN_DATASET_FILES:
	# 	h5_filepath = os.path.join(TRAIN_DATASET_DIR, datasetFile)
	# 	dataset = DatasetFromHdf5(h5_filepath)
	# 	train_data = dataset
	# 	print("Length of Training Set (%s):" % datasetFile, len(dataset))

	# for datasetFile in VALID_DATASET_FILES:
	# 	h5_filepath = os.path.join(TRAIN_DATASET_DIR, datasetFile)
	# 	dataset = DatasetFromHdf5(h5_filepath)
	# 	valid_data = dataset
	# 	print("Length of Validation Set (%s):" % datasetFile, len(dataset))

	for test_dataset in TEST_DATASETS:
		for illumination in ILLUMINATIONS:
			dataset_dir = os.path.join(TEST_ROOT_DATASET_DIR, "working_%s" % test_dataset, "%s_%s_204ch" % (test_dataset, illumination), "train", "cameraRGBN")
			dataset = DatasetFromDirectory(dataset_dir)
			train_data = dataset
			print("Length of Training Set (%s):" % test_dataset, len(dataset))

	for test_dataset in TEST_DATASETS:
		for illumination in ILLUMINATIONS:
			dataset_dir = os.path.join(TEST_ROOT_DATASET_DIR, "working_%s" % test_dataset, "%s_%s_204ch" % (test_dataset, illumination), "valid", "cameraRGBN")
			dataset = DatasetFromDirectory(dataset_dir)
			valid_data = dataset
			print("Length of Validation Set (%s):" % test_dataset, len(dataset))

	# Data Loader (Input Pipeline)
	train_data_loader = DataLoader(dataset=ConcatDataset([train_data]),
								   num_workers=1,
								   batch_size=batch_size,
								   shuffle=True,
								   pin_memory=True)

	val_data_loader = DataLoader(dataset=ConcatDataset([valid_data]),
								   num_workers=1,
								   batch_size=1,
								   shuffle=True,
								   pin_memory=True)
	print("Data Loaded")
	# Parameters, Loss and Optimizer
	start_epoch = 0
	iteration = 0
	criterion_mrae = mrae_loss
	criterion_sam = sam_loss

	logger = initialize_logger(filename="train.log")
	loss_csv = open(os.path.join(LOGS_PATH, "loss.csv"), "w+")

	# Resume
	# resume_file = ""
	# if resume_file:
	# 	if os.path.isfile(resume_file):
	# 		print("=> loading checkpoint '{}'".format(resume_file))
	# 		checkpoint = torch.load(resume_file)
	# 		start_epoch = checkpoint["epoch"]
	# 		iteration = checkpoint["iter"]
	# 		model.load_state_dict(checkpoint["state_dict"])
	# 		optimizer.load_state_dict(checkpoint["optimizer"])

	log_string = "Epoch [%2d], Iter[%4d], Time:%.9f, Learning Rate: %.9f, Train Loss: %.9f (%.9f, %.9f), Validation Loss: %.9f (%.9f, %.9f)"
	fileprestring = "%s_%s" % (MODEL_NAME, DATASET_NAME)

	for fusion in fusion_techniques:
		model = Network(ResNeXtBottleneck, block_num=10, input_channel=4, output_channel=51, fusion=fusion)
		optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
		# print(summary(model, (4, 64, 64), verbose=1))
		if torch.cuda.device_count() > 1:
			model = nn.DataParallel(model)
		if torch.cuda.is_available():
			model.cuda()

		model_run = model_run_title % (fusion, MODEL_NAME, DATASET_NAME)

		print("\n" + model_run)
		logger.info(model_run)

		for epoch in range(start_epoch+1, end_epoch):
			start_time = time.time()

			train_loss, train_loss_mrae, train_loss_sam, iteration, lr = train(train_data_loader, model, criterion_mrae, criterion_sam, optimizer, iteration, init_lr, end_epoch)
			val_loss, val_loss_mrae, val_loss_sam = validate(val_data_loader, model, criterion_mrae, criterion_sam)

			save_checkpoint(epoch, fileprestring, fusion, iteration, model, optimizer)

			end_time = time.time()
			epoch_time = end_time - start_time

			# Printing and saving losses
			record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, val_loss)
			print(log_string % (epoch, iteration, epoch_time, lr, train_loss, train_loss_mrae, train_loss_sam, val_loss, val_loss_mrae, val_loss_sam))
			logger.info(log_string % (epoch, iteration, epoch_time, lr, train_loss, train_loss_mrae, train_loss_sam, val_loss, val_loss_mrae, val_loss_sam))
		iteration = 0

# Training
def train(train_data_loader, model, criterion_mrae, criterion_sam, optimizer, iteration, init_lr, end_epoch):
	""" Trains the model on the dataloader provided """
	model.train()
	losses = AverageMeter()
	losses_mrae, losses_sam = AverageMeter(), AverageMeter()

	for i, (images, labels) in enumerate(train_data_loader):
		labels = labels.cuda()
		images = images.cuda()

		images = Variable(images)
		labels = Variable(labels)

		lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=968000, power=1.5)
		iteration = iteration + 1

		# Forward + Backward + Optimize
		output = model(images)
		loss_mrae = criterion_mrae(output, labels).cuda()
		loss_sam = criterion_sam(output, labels).cuda()

		loss = loss_mrae + loss_sam * 0.1
		loss = loss.cuda()

		# print(i, loss_mrae, loss_sam, loss)

		optimizer.zero_grad()
		loss.backward()

		# Calling the step function on an Optimizer makes an update to its parameters
		optimizer.step()
		#  record loss
		losses.update(loss.item())
		losses_mrae.update(loss_mrae.item())
		losses_sam.update(loss_sam.item())

	return losses.avg, losses_mrae.avg, losses_sam.avg, iteration, lr

# Validate
def validate(val_data_loader, model, criterion_mrae, criterion_sam):
	""" Validates the model on the dataloader provided """
	model.eval()
	losses = AverageMeter()
	losses_mrae, losses_sam = AverageMeter(), AverageMeter()

	for i, (images, labels) in enumerate(val_data_loader):
		images = images.cuda()
		labels = labels.cuda()
		with torch.no_grad():
			images = torch.autograd.Variable(images)
			labels = torch.autograd.Variable(labels)

		# compute output
		output = model(images)
		loss_mrae = criterion_mrae(output, labels).cuda()
		loss_sam = criterion_sam(output, labels).cuda()

		loss = loss_mrae + loss_sam * 0.1
		loss = loss.cuda()

		#  record loss
		losses.update(loss.item())
		losses_mrae.update(loss_mrae.item())
		losses_sam.update(loss_sam.item())

	return losses.avg, losses_mrae.avg, losses_sam.avg

# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1,
					  max_iter=100, power=0.9):
	"""Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power
	"""
	if iteraion % lr_decay_iter or iteraion > max_iter:
		return optimizer

	lr = init_lr*(1 - iteraion/max_iter)**power

	for param_group in optimizer.param_groups:
		param_group["lr"] = lr

	return lr

if __name__ == "__main__":
	init_directories()
	# make_h5_dataset(DATASET_DIR=os.path.join(os.path.dirname(TRAIN_DATASET_DIR), "working_datasets", "working_avocado", "avocado_h_204ch", "train", "cameraRGBN"), h5_filename="train_avocado_halogen_4to51bands_whole.h5")
	# make_h5_dataset(DATASET_DIR=os.path.join(os.path.dirname(TRAIN_DATASET_DIR), "working_datasets", "working_avocado", "avocado_h_204ch", "valid", "cameraRGBN"), h5_filename="valid_avocado_halogen_4to51bands_whole.h5")
	main()
	# modeltoONNX()
	# ONNXtotf()
	# tf_to_tflite()