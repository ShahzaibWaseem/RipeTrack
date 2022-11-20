from __future__ import division

import  os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torchsummary import summary
from torch.autograd import Variable

from models.model import Network
from models.resblock import ResNeXtBottleneck
from loss import mrae_loss, sam_loss, sid_loss, weighted_loss

from dataset import get_dataloaders, get_required_transforms
from utils import AverageMeter, initialize_logger, save_checkpoint, get_best_checkpoint, record_loss, poly_lr_scheduler, makeMobileModel, make_h5_dataset, modeltoONNX, ONNXtotf, tf_to_tflite
from config import LOGS_PATH, BANDS, init_directories, batch_size, device, end_epoch, init_lr, model_run_title, run_pretrained, predef_input_transform, predef_label_transform

torch.autograd.set_detect_anomaly(True)

def main():
	torch.backends.cudnn.benchmark = True
	trainset_size=0.8
	# input_transform, label_transform = get_required_transforms()
	input_transform, label_transform = predef_input_transform, predef_label_transform
	train_data_loader, val_data_loader, whole_dataset_loader = get_dataloaders(input_transform, label_transform, task="reconstruction", trainset_size=trainset_size)
	# train_data_loader, val_data_loader = train_data_loader.to(device), val_data_loader.to(device)

	# Parameters, Loss and Optimizer
	start_epoch = 0
	iteration = 0
	criterion_mrae = mrae_loss
	criterion_sam = sam_loss
	criterion_sid = sid_loss
	criterion_weighted = weighted_loss

	criterions = (criterion_mrae, criterion_sam, criterion_sid, criterion_weighted)

	# Log files
	logger = initialize_logger(filename="train.log")
	loss_csv = open(os.path.join(LOGS_PATH, "loss.csv"), "w+")

	log_string = "Epoch [%3d], Iter[%5d], Time: %.9f, Learning Rate: %.9f, Train Loss: %.9f (%.9f, %.9f, %.9f), Validation Loss: %.9f (%.9f, %.9f, %.9f)"

	# make model
	model = Network(block=ResNeXtBottleneck, block_num=10, input_channel=4, n_hidden=64, output_channel=len(BANDS))
	optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
	# print(summary(model, (4, 64, 64), verbose=1))

	if run_pretrained:
		epoch, iter, state_dict, optimizer, val_loss, val_acc = get_best_checkpoint(task="reconstruction")
		model.load_state_dict(state_dict)
		optimizer.load_state_dict(optimizer)

	# # Resume
	# resume_file = checkpoint_file
	# if resume_file:
	# 	if os.path.isfile(resume_file):
	# 		print("=> loading checkpoint '{}'".format(resume_file))
	# 		checkpoint = torch.load(resume_file)
	# 		start_epoch = checkpoint["epoch"]
	# 		iteration = checkpoint["iter"]
	# 		model.load_state_dict(checkpoint["state_dict"])
	# 		optimizer.load_state_dict(checkpoint["optimizer"])

	# Multi Device Cuda
	model.to(device)

	print("\n" + model_run_title)
	logger.info(model_run_title)

	for epoch in range(start_epoch+1, end_epoch):
		start_time = time.time()

		train_loss, train_losses_ind, iteration, lr = train(train_data_loader, model, criterions, optimizer, iteration, init_lr, int(trainset_size*len(whole_dataset_loader))*end_epoch/batch_size)
		val_loss, val_losses_ind = validate(val_data_loader, model, criterions)

		train_loss_mrae, train_loss_sam, train_loss_sid = train_losses_ind
		val_loss_mrae, val_loss_sam, val_loss_sid = val_losses_ind

		save_checkpoint(epoch, iteration, model, optimizer, val_loss, 0, task="reconstruction")
		epoch_time = time.time() - start_time

		# Printing and saving losses
		record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, val_loss)
		log_string_filled = log_string % (epoch, iteration, epoch_time, lr,
							train_loss, train_loss_mrae, train_loss_sam, train_loss_sid,
							val_loss, val_loss_mrae, val_loss_sam, val_loss_sid)

		print("\n", log_string_filled, "\n")
		logger.info(log_string_filled)
	iteration = 0

def train(train_data_loader, model, criterions, optimizer, iteration, init_lr, max_iter):
	""" Trains the model on the dataloader provided """
	model.train()
	losses = AverageMeter()
	criterion_mrae, criterion_sam, criterion_sid, criterion_weighted = criterions
	losses_mrae, losses_sam, losses_sid, losses_weighted = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

	for images, labels, _, _ in tqdm(train_data_loader, desc="Train", total=len(train_data_loader)):
		# print(torch.min(images), torch.max(images), torch.min(labels), torch.max(labels))
		images, labels = Variable(images.to(device, non_blocking=True)), Variable(labels.to(device, non_blocking=True))

		lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=max_iter, power=0.9)
		iteration = iteration + 1

		# Forward + Backward + Optimize
		output = model(images)

		loss_mrae = criterion_mrae(output, labels)
		loss_sam = torch.mul(criterion_sam(output, labels), 0.1)
		loss_sid = torch.mul(criterion_sid(output, labels), 0.0001)
		# loss_weighted = criterion_weighted(output, labels)
		loss = loss_mrae.add_(loss_sam).add_(loss_sid)
		# loss = loss_mrae + loss_sam + loss_sid

		optimizer.zero_grad()
		loss.backward()

		nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
		# Calling the step function on an Optimizer makes an update to its parameters
		optimizer.step()
		#  record loss
		losses.update(loss.item())
		losses_mrae.update(loss_mrae.item())
		losses_sam.update(loss_sam.item())
		losses_sid.update(loss_sid.item())
		# losses_weighted.update(loss_weighted.item())

	return losses.avg, (losses_mrae.avg, losses_sam.avg, losses_sid.avg), iteration, lr

def validate(val_data_loader, model, criterions):
	""" Validates the model on the dataloader provided """
	model.eval()
	losses = AverageMeter()
	criterion_mrae, criterion_sam, criterion_sid, criterion_weighted = criterions
	losses_mrae, losses_sam, losses_sid, losses_weighted = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

	for images, labels, _, _ in tqdm(val_data_loader, desc="Valid", total=len(val_data_loader)):
		images = images.cuda()
		labels = labels.cuda()

		with torch.no_grad():
			images = torch.autograd.Variable(images)
			labels = torch.autograd.Variable(labels)

		# compute output
		output = model(images)

		loss_mrae = criterion_mrae(output, labels)
		loss_sam = torch.mul(criterion_sam(output, labels), 0.1)
		loss_sid = torch.mul(criterion_sid(output, labels), 0.0001)
		# loss_weighted = criterion_weighted(output, labels)
		loss = loss_mrae.add_(loss_sam).add_(loss_sid)
		# loss = loss_mrae + loss_sam + loss_sid

		#  record loss
		losses.update(loss.item())
		losses_mrae.update(loss_mrae.item())
		losses_sam.update(loss_sam.item())
		losses_sid.update(loss_sid.item())
		# losses_weighted.update(loss_weighted.item())

	return losses.avg, (losses_mrae.avg, losses_sam.avg, losses_sid.avg)

if __name__ == "__main__":
	init_directories()
	# makeMobileModel()
	# make_h5_dataset(TRAIN_DATASET_DIR=os.path.join(TRAIN_DATASET_DIR, "train"), h5_filename="train_apple_halogen_4to51bands_whole.h5")
	# make_h5_dataset(TRAIN_DATASET_DIR=os.path.join(TRAIN_DATASET_DIR, "valid"), h5_filename="valid_apple_halogen_4to51bands_whole.h5")
	main()
	# modeltoONNX()
	# ONNXtotf()
	# tf_to_tflite()