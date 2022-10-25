from __future__ import division

import  os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset, random_split

from models.model import Network
from models.resblock import resblock, ResNeXtBottleneck
from loss import mrae_loss, sam_loss, sid_loss, weighted_loss
from dataset import DatasetFromDirectory, DatasetFromHdf5, get_normalization_parameters

from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, makeMobileModel, make_h5_dataset, modeltoONNX, ONNXtotf, tf_to_tflite
from config import TEST_ROOT_DATASET_DIR, TRAIN_DATASET_DIR, TRAIN_DATASET_FILES, VALID_DATASET_FILES, LOGS_PATH, MODEL_NAME, DATASET_NAME, NUMBER_OF_BANDS, BANDS, PATCH_SIZE, init_directories, checkpoint_file, batch_size, end_epoch, init_lr, model_run_title

torch.autograd.set_detect_anomaly(True)

def get_required_transforms(task="reconstruction"):
	""" Returns the two transforms for the RGB-NIR image input and Hypercube label.
		Note: The `dataset` recieved is already Tensor data.
		This function gets the dataset specified in the `config.py` file. """
	# Dataset
	image_mean, image_std, hypercube_mean, hypercube_std = 0, 0, 0, 0

	dataset = DatasetFromDirectory(root=TEST_ROOT_DATASET_DIR,
								   dataset_name=DATASET_NAME,
								   task=task,
								   patch_size=PATCH_SIZE,
								   lazy_read=False,
								   shuffle=False,
								   rgbn_from_cube=False,
								   use_all_bands=False if task == "reconstruction" else True,
								   product_pairing=False,
								   train_with_patches=True,
								   positive_only=False,
								   verbose=False,
								   augment_factor=0,
								   transform=(None, None))

	dataloader = DataLoader(dataset=dataset,
							num_workers=1,
							batch_size=batch_size,
							shuffle=False,
							pin_memory=True)

	(image_mean, image_std), (hypercube_mean, hypercube_std) = get_normalization_parameters(dataloader)

	print(75*"-" + "\nDataset Normalization\n" + 75*"-")

	if task == "reconstruction":
		print("RGB-NIR Images Size:\t\t\t\t\t%d" % (image_mean.size(dim=0)))
		print("The Mean of the dataset is in the range:\t\t%f - %f"
			% (torch.min(image_mean).item(), torch.max(image_mean).item()))
		print("The Standard Deviation of the dataset is in the range:\t%f - %f\n"
			% (torch.min(image_std).item(), torch.max(image_std).item()))

	print("Hypercubes Size:\t\t\t\t\t%d" % (hypercube_std.size(dim=0)))
	print("The Mean of the dataset is in the range:\t\t%f - %f"
		  % (torch.min(hypercube_mean).item(), torch.max(hypercube_mean).item()))
	print("The Standard Deviation of the dataset is in the range:\t%f - %f"
		  % (torch.min(hypercube_std).item(), torch.max(hypercube_std).item()))
	print(75*"-")
	del dataset, dataloader

	hypercube_transform_list = []

	if task == "classification":
		hypercube_transform_list = [# transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
									# transforms.RandomResizedCrop(size=(PATCH_SIZE, PATCH_SIZE)),
									# transforms.CenterCrop(size=(PATCH_SIZE, PATCH_SIZE)),
									transforms.RandomRotation(20, interpolation=transforms.InterpolationMode.BILINEAR),
									transforms.RandomHorizontalFlip()]

	# Data is already tensor, so just normalize it
	hypercube_transform_list.append(transforms.Normalize(mean=hypercube_mean, std=hypercube_std))
	input_transform = transforms.Compose([transforms.Normalize(mean=image_mean, std=image_std)]) if task == "reconstruction" else None
	hypercube_transform = transforms.Compose([transforms.Normalize(mean=hypercube_mean, std=hypercube_std)])

	del hypercube_transform_list


	return input_transform, hypercube_transform

def main():
	torch.backends.cudnn.benchmark = True

	input_transform, label_transform = get_required_transforms()
	dataset = DatasetFromDirectory(root=TEST_ROOT_DATASET_DIR,
								   dataset_name=DATASET_NAME,
								   task="reconstruction",
								   patch_size=PATCH_SIZE,
								   lazy_read=False,
								   shuffle=True,
								   rgbn_from_cube=False,
								   use_all_bands=False,
								   product_pairing=False,
								   train_with_patches=True,
								   positive_only=True,
								   verbose=True,
								   transform=(input_transform, label_transform))

	trainset_size = 0.8
	print("Dataset size:\t\t\t{}".format(len(dataset)))

	train_data, valid_data = random_split(dataset, [int(trainset_size*len(dataset)), len(dataset) - int(len(dataset)*trainset_size)])
	print("Length of Training Set ({}%):\t{}".format(round(trainset_size * 100), len(train_data)))
	print("Length of Validation Set ({}%):\t{}".format(round((1-trainset_size) * 100), len(valid_data)))

	# train_data, valid_data = [], []
	# for datasetFile in TRAIN_DATASET_FILES:
	# 	h5_filepath = os.path.join(TRAIN_DATASET_DIR, datasetFile)
	# 	dataset = DatasetFromHdf5(h5_filepath)
	# 	train_data.append(dataset)
	# 	print("Length of Training Set (%s):" % datasetFile, len(dataset))

	# for datasetFile in VALID_DATASET_FILES:
	# 	h5_filepath = os.path.join(TRAIN_DATASET_DIR, datasetFile)
	# 	dataset = DatasetFromHdf5(h5_filepath)
	# 	valid_data.append(dataset)
	# 	print("Length of Validation Set (%s):" % datasetFile, len(dataset))

	# Data Loader (Input Pipeline)
	train_data_loader = DataLoader(dataset=train_data,
								   num_workers=1,
								   batch_size=batch_size,
								   shuffle=True,
								   pin_memory=True)

	val_data_loader = DataLoader(dataset=valid_data,
								 num_workers=1,
								 batch_size=2,
								 shuffle=False,
								 pin_memory=True)

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
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	if torch.cuda.is_available():
		model.cuda()

	print("\n" + model_run_title)
	logger.info(model_run_title)

	for epoch in range(start_epoch+1, end_epoch):
		start_time = time.time()

		train_loss, train_losses_ind, iteration, lr = train(train_data_loader, model, criterions, optimizer, iteration, init_lr, int(trainset_size*len(dataset))*end_epoch/batch_size)
		val_loss, val_losses_ind = validate(val_data_loader, model, criterions)

		train_loss_mrae, train_loss_sam, train_loss_sid = train_losses_ind
		val_loss_mrae, val_loss_sam, val_loss_sid = val_losses_ind

		save_checkpoint(epoch, iteration, model, optimizer)
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

	for images, labels, _ in tqdm(train_data_loader, desc="Train", total=len(train_data_loader)):
		# print(torch.min(images), torch.max(images), torch.min(labels), torch.max(labels))
		labels = labels.cuda()
		images = images.cuda()

		images = Variable(images)
		labels = Variable(labels)
		lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=968000, power=0.9)
		iteration = iteration + 1

		# Forward + Backward + Optimize
		output = model(images)

		loss_mrae = criterion_mrae(output, labels)
		# print(output, labels, loss_mrae)
		loss_sam = criterion_sam(output, labels) * 0.1
		loss_sid = criterion_sid(output, labels) * 0.0001
		# loss_weighted = criterion_weighted(output, labels)

		loss = loss_mrae + loss_sam + loss_sid

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

	for images, labels, _ in tqdm(val_data_loader, desc="Valid", total=len(val_data_loader)):
		images = images.cuda()
		labels = labels.cuda()

		with torch.no_grad():
			images = torch.autograd.Variable(images)
			labels = torch.autograd.Variable(labels)

		# compute output
		output = model(images)

		loss_mrae = criterion_mrae(output, labels)
		loss_sam = criterion_sam(output, labels) * 0.1
		loss_sid = criterion_sid(output, labels) * 0.0001
		# loss_weighted = criterion_weighted(output, labels)

		loss = loss_mrae + loss_sam + loss_sid

		#  record loss
		losses.update(loss.item())
		losses_mrae.update(loss_mrae.item())
		losses_sam.update(loss_sam.item())
		losses_sid.update(loss_sid.item())
		# losses_weighted.update(loss_weighted.item())

	return losses.avg, (losses_mrae.avg, losses_sam.avg, losses_sid.avg)

def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9):
	"""
	Polynomial decay of learning rate
		init_lr:		base learning rate
		iter:			current iteration
		lr_decay_iter:	how frequently decay occurs, default is 1
		max_iter:		number of maximum iterations
		power:			polymomial power
	"""
	if iteraion % lr_decay_iter or iteraion > max_iter:
		return optimizer

	lr = init_lr * (1 - iteraion/max_iter) ** power

	for param_group in optimizer.param_groups:
		param_group["lr"] = lr

	return lr

if __name__ == "__main__":
	# init_directories()
	# makeMobileModel()
	# make_h5_dataset(TRAIN_DATASET_DIR=os.path.join(TRAIN_DATASET_DIR, "train"), h5_filename="train_apple_halogen_4to51bands_whole.h5")
	# make_h5_dataset(TRAIN_DATASET_DIR=os.path.join(TRAIN_DATASET_DIR, "valid"), h5_filename="valid_apple_halogen_4to51bands_whole.h5")
	main()
	# modeltoONNX()
	# ONNXtotf()
	# tf_to_tflite()