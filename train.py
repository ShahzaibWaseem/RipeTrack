from __future__ import division

import  os
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torchsummary import summary
from torch.autograd import Variable

from models.MST import MST_Plus_Plus
from loss import mrae_loss, sam_loss, sid_loss

from dataset import get_required_transforms, get_dataloaders_reconstruction
from utils import AverageMeter, initialize_logger, save_checkpoint, get_best_checkpoint, record_loss, poly_lr_scheduler, optimizer_to
from config import MODEL_PATH, LOGS_PATH, DATA_PREP_PATH, BANDS, PREDEF_TRANSFORMS_FILENAME, batch_size, device, end_epoch, init_lr, model_run_title, run_pretrained, lossfunctions_considered, create_directory

torch.autograd.set_detect_anomaly(False)

parser = argparse.ArgumentParser()
parser.add_argument("--disable_tqdm", default=False, required=False, type=bool, help="Disable tqdm progress bar")
args = parser.parse_args()
disable_tqdm = args.disable_tqdm

def get_and_save_predef_transforms():
	if os.path.exists(os.path.join(DATA_PREP_PATH, PREDEF_TRANSFORMS_FILENAME)):
		print("Loading Pre-defined Transforms...")
		transforms = torch.load(os.path.join(DATA_PREP_PATH, PREDEF_TRANSFORMS_FILENAME))
		input_transform, label_transform = transforms["rgbn"], transforms["hypercube"]
		if BANDS == transforms["bands"]:
			return input_transform, label_transform
		else:
			input_transform, label_transform = get_required_transforms()
			transforms = {"rgbn": input_transform, "hypercube": label_transform, "bands": BANDS}
			torch.save(transforms, os.path.join(DATA_PREP_PATH, PREDEF_TRANSFORMS_FILENAME))
			return input_transform, label_transform
	else:
		input_transform, label_transform = get_required_transforms()
		transforms = {"rgbn": input_transform, "hypercube": label_transform, "bands": BANDS}
		torch.save(transforms, os.path.join(DATA_PREP_PATH, PREDEF_TRANSFORMS_FILENAME))
		return input_transform, label_transform

def main():
	# torch.backends.cudnn.benchmark = True
	trainset_size=0.8
	# input_transform, label_transform = get_and_save_predef_transforms()

	# train_data_loader, valid_data_loader, whole_dataset_loader = get_dataloaders(input_transform, label_transform, task="reconstruction", trainset_size=trainset_size)
	# train_data_loader, valid_data_loader = train_data_loader.to(device), valid_data_loader.to(device)

	train_data_loader, valid_data_loader = get_dataloaders_reconstruction(trainset_size=trainset_size)

	whole_dataset_size = len(train_data_loader.dataset) + len(valid_data_loader.dataset)

	# Parameters, Loss and Optimizer
	start_epoch = 0
	iteration = 0
	best_val_loss = 0
	criterion_mrae = mrae_loss
	criterion_sam = sam_loss
	criterion_sid = sid_loss

	criterions = (criterion_mrae, criterion_sam, criterion_sid)

	# Log files
	logger = initialize_logger(filename="train.log")

	log_string = "Epoch [%3d], Iter[%6d], Time: %.9f, Learning Rate: %.9f, Train Loss: %.9f (%.9f, %.9f, %.9f), Validation Loss: %.9f (%.9f, %.9f, %.9f)"

	# make model
	model = MST_Plus_Plus(in_channels=4, out_channels=len(BANDS), n_feat=len(BANDS), stage=3)
	optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
	# print(summary(model, (4, 64, 64), verbose=1))

	if run_pretrained:
		best_checkpoint_file, epoch, iter, state_dict, opt_state, val_loss, val_acc = get_best_checkpoint(task="reconstruction")
		model.load_state_dict(state_dict)
		optimizer.load_state_dict(opt_state)
		start_epoch = epoch

	model.to(device, non_blocking=True)
	optimizer_to(optimizer, device)

	print("\n" + model_run_title)
	logger.info(model_run_title)

	for epoch in range(start_epoch+1, end_epoch):
		start_time = time.time()

		train_loss, train_losses_ind, iteration, lr = train(train_data_loader, model, criterions, optimizer, iteration, init_lr, int(trainset_size*whole_dataset_size)*end_epoch/batch_size)
		val_loss, val_losses_ind = validate(valid_data_loader, model, criterions)

		train_loss_mrae, train_loss_sam, train_loss_sid = train_losses_ind
		val_loss_mrae, val_loss_sam, val_loss_sid = val_losses_ind
		if best_val_loss < val_loss:
			best_val_loss = val_loss
			best_epoch = epoch
			best_model = model
			best_optimizer = optimizer
			iteration_passed = iteration
		# if epoch % 30 == 0:
		save_checkpoint(epoch, iteration, model, optimizer, val_loss, 0, 0, bands=BANDS, task="reconstruction")

		epoch_time = time.time() - start_time

		# Printing and saving losses
		# record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, val_loss)
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
	criterion_mrae, criterion_sam, criterion_sid = criterions
	losses_mrae, losses_sam, losses_sid = AverageMeter(), AverageMeter(), AverageMeter()

	for images, labels in tqdm(train_data_loader, desc="Train", total=len(train_data_loader), disable=disable_tqdm):
		# print(torch.min(images), torch.max(images), torch.min(labels), torch.max(labels))
		images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
		lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=max_iter, power=0.9)
		iteration = iteration + 1

		# Forward + Backward + Optimize
		output = model(images)

		loss_mrae = criterion_mrae(output, labels)
		loss_sam = torch.mul(criterion_sam(output, labels), 0.1) if "SAM" in lossfunctions_considered else torch.tensor(0)
		loss_sid = torch.mul(criterion_sid(output, labels), 0.0001) if "SID" in lossfunctions_considered else torch.tensor(0)
		loss = loss_mrae + loss_sam + loss_sid

		optimizer.zero_grad()
		loss.backward()

		nn.utils.clip_grad_norm_(model.parameters(), 5.0)
		# Calling the step function on an Optimizer makes an update to its parameters
		optimizer.step()
		#  record loss
		losses.update(loss.item())
		losses_mrae.update(loss_mrae.item())
		losses_sam.update(loss_sam.item())
		losses_sid.update(loss_sid.item())

	return losses.avg, (losses_mrae.avg, losses_sam.avg, losses_sid.avg), iteration, lr

def validate(valid_data_loader, model, criterions):
	""" Validates the model on the dataloader provided """
	model.eval()
	losses = AverageMeter()
	criterion_mrae, criterion_sam, criterion_sid = criterions
	losses_mrae, losses_sam, losses_sid = AverageMeter(), AverageMeter(), AverageMeter()

	with torch.no_grad():
		for images, labels in tqdm(valid_data_loader, desc="Valid", total=len(valid_data_loader), disable=disable_tqdm):
			images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

			# compute output
			output = model(images)

			loss_mrae = criterion_mrae(output, labels)
			loss_sam = torch.mul(criterion_sam(output, labels), 0.1) if "SAM" in lossfunctions_considered else torch.tensor(0)
			loss_sid = torch.mul(criterion_sid(output, labels), 0.0001) if "SID" in lossfunctions_considered else torch.tensor(0)
			loss = loss_mrae + loss_sam + loss_sid

			#  record loss
			losses.update(loss.item())
			losses_mrae.update(loss_mrae.item())
			losses_sam.update(loss_sam.item())
			losses_sid.update(loss_sid.item())

	return losses.avg, (losses_mrae.avg, losses_sam.avg, losses_sid.avg)

if __name__ == "__main__":
	create_directory(os.path.join(MODEL_PATH, "reconstruction"))
	create_directory(LOGS_PATH)
	main()