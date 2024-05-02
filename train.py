from __future__ import division

import  os
import time
import argparse
from tqdm import tqdm

import torch
from torchsummary import summary

from models.MST import MST_Plus_Plus
from loss import Loss_MRAE, Loss_SAM, Loss_SID

from dataset import get_dataloaders_reconstruction
from utils import AverageMeter, create_directory, initialize_logger, save_checkpoint, get_best_checkpoint, optimizer_to
from config import MODEL_PATH, LOGS_PATH, BANDS,\
	batch_size, device, end_epoch, init_lr, lossfunctions_considered, model_run_title, run_pretrained, transfer_learning

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.autograd.set_detect_anomaly(False)

parser = argparse.ArgumentParser()
parser.add_argument("--disable_tqdm", default=False, required=False, type=bool, help="Disable tqdm progress bar")
args = parser.parse_args()
disable_tqdm = args.disable_tqdm

def main():
	torch.backends.cudnn.benchmark = True
	train_data_loader, valid_data_loader = get_dataloaders_reconstruction()
	whole_dataset_size = len(train_data_loader.dataset) + len(valid_data_loader.dataset)
	# train_data_loader, valid_data_loader = train_data_loader.to(device), valid_data_loader.to(device)

	# Parameters, Loss and Optimizer
	start_epoch = 0
	iteration = 0
	best_val_loss = float("inf")
	criterion_mrae = Loss_MRAE()
	criterion_sam = Loss_SAM()
	criterion_sid = Loss_SID()

	criterion_mrae.to(device)
	criterion_sam.to(device)
	criterion_sid.to(device)

	criterions = (criterion_mrae, criterion_sam, criterion_sid)

	# Log files
	logger = initialize_logger(filename="train.log")
	log_string = "Epoch [%3d], Iter[%7d], Time: %.9f, Learning Rate: %.9f, Train Loss: %.9f (%.9f, %.9f, %.9f)"
	log_string_val = "Validation Loss: %.9f (%.9f, %.9f, %.9f)"

	# make model
	model = MST_Plus_Plus(in_channels=4, out_channels=len(BANDS), n_feat=len(BANDS)//2, msab_stages=2, stage=1)
	optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(train_data_loader.__len__())*end_epoch/batch_size, eta_min=1e-6)
	summary(model, (4, 512, 512), verbose=1)

	if run_pretrained:
		# checkpoint_filename, epoch, iter, state_dict, optimizer, val_loss, val_acc = get_best_checkpoint(task="reconstruction")
		checkpoint_filename = "RT_MST++_shelflife_080 RGBNIR Final [ThinModel][L+A].pkl"
		checkpoint = torch.load(os.path.join(MODEL_PATH, "reconstruction", "others", checkpoint_filename))
		epoch, iter, state_dict, optimizer, val_loss, val_acc = checkpoint["epoch"], checkpoint["iter"], checkpoint["state_dict"],\
			checkpoint["optimizer"], checkpoint["val_loss"], checkpoint["val_acc"]
		model.load_state_dict(state_dict)
		optimizer.load_state_dict(optimizer)
		start_epoch = epoch
		print("Loaded model from checkpoint: Filename: %s Epochs Run: %d, Validation Loss: %.9f" % (checkpoint_filename, epoch, val_loss))

	if transfer_learning:
		module_count = 0
		for param in model.parameters():
			param.requires_grad = False
		for _, p in model.state_dict().items():
			module_count += 1
			if module_count > 151:		# 151 layers onwards are parameters in the last MST block
				p.requires_grad = True
			# print(_, p.requires_grad)
		# print("Total number of modules: ", module_count)

	model.to(device)
	# optimizer_to(optimizer, device)

	print("\n" + model_run_title)
	logger.info(model_run_title)

	for epoch in range(start_epoch+1, end_epoch):
		# torch.cuda.synchronize()
		start_time = time.time()

		train_loss, train_losses_ind, iteration, lr = train(train_data_loader, model, criterions, optimizer, iteration, scheduler)
		train_loss_mrae, train_loss_sam, train_loss_sid = train_losses_ind

		if epoch % 20 == 0:
			val_loss, val_losses_ind = validate(valid_data_loader, model, criterions)
			val_loss_mrae, val_loss_sam, val_loss_sid = val_losses_ind
			if best_val_loss > val_loss:
				best_val_loss = val_loss
				best_epoch = epoch
				best_model = model
				best_optimizer = optimizer
				iteration_passed = iteration
			save_checkpoint(int(round(epoch, -1)), iteration_passed, best_model, best_optimizer, best_val_loss, 0, 0, bands=BANDS, task="reconstruction")
			log_string_val_filled = log_string_val % (val_loss, val_loss_mrae, val_loss_sam, val_loss_sid)
			print("\n" + log_string_val_filled + "\n")
			logger.info(log_string_val_filled)

		# torch.cuda.synchronize()
		epoch_time = time.time() - start_time

		# Printing and saving losses
		log_string_filled = log_string % (epoch, iteration, epoch_time, lr,
							train_loss, train_loss_mrae, train_loss_sam, train_loss_sid)

		print("\n"+ log_string_filled +"\n")
		logger.info(log_string_filled)
	iteration = 0

def train(train_data_loader, model, criterions, optimizer, iteration, scheduler):
	""" Trains the model on the dataloader provided """
	model.train()
	criterion_mrae, criterion_sam, criterion_sid = criterions
	losses, losses_mrae, losses_sam, losses_sid = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

	for images, labels in tqdm(train_data_loader, desc="Train", total=len(train_data_loader), disable=disable_tqdm):
		# print(torch.min(images), torch.max(images), torch.min(labels), torch.max(labels))
		images, labels = images.to(device), labels.to(device)
		lr = optimizer.param_groups[0]["lr"]
		iteration += 1

		# Forward + Backward + Optimize
		optimizer.zero_grad()
		output = model(images)

		loss_mrae = criterion_mrae(output, labels)
		loss_sam = torch.mul(criterion_sam(output, labels), 0.1) if "SAM" in lossfunctions_considered else torch.tensor(0)
		loss_sid = torch.mul(criterion_sid(output, labels), 0.0001) if "SID" in lossfunctions_considered else torch.tensor(0)
		loss = loss_mrae + loss_sam + loss_sid

		loss.backward()

		# Calling the step function on an Optimizer makes an update to its parameters
		optimizer.step()
		scheduler.step()
		# record loss
		losses.update(loss.item())
		losses_mrae.update(loss_mrae.item())
		losses_sam.update(loss_sam.item())
		losses_sid.update(loss_sid.item())

	return losses.avg, (losses_mrae.avg, losses_sam.avg, losses_sid.avg), iteration, lr

def validate(valid_data_loader, model, criterions):
	""" Validates the model on the dataloader provided """
	model.eval()
	criterion_mrae, criterion_sam, criterion_sid = criterions
	losses, losses_mrae, losses_sam, losses_sid = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

	with torch.no_grad():
		for images, labels in tqdm(valid_data_loader, desc="Valid", total=len(valid_data_loader), disable=disable_tqdm):
			images, labels = images.to(device), labels.to(device)

			# compute output
			output = model(images)

			loss_mrae = criterion_mrae(output, labels)
			loss_sam = torch.mul(criterion_sam(output, labels), 0.1) if "SAM" in lossfunctions_considered else torch.tensor(0)
			loss_sid = torch.mul(criterion_sid(output, labels), 0.0001) if "SID" in lossfunctions_considered else torch.tensor(0)
			loss = loss_mrae + loss_sam + loss_sid

			# record loss
			losses.update(loss.item())
			losses_mrae.update(loss_mrae.item())
			losses_sam.update(loss_sam.item())
			losses_sid.update(loss_sid.item())

	return losses.avg, (losses_mrae.avg, losses_sam.avg, losses_sid.avg)

if __name__ == "__main__":
	create_directory(os.path.join(MODEL_PATH, "reconstruction"))
	create_directory(LOGS_PATH)
	main()