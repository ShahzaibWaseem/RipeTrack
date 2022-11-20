from __future__ import division

import os
import json
import numpy as np

import h5py
import hdf5storage

import logging
from glob import glob
from imageio import imread

import torch
from torch.autograd import Variable
from torch.utils.mobile_optimizer import optimize_for_mobile

from models.model import Network
from models.resblock import ResNeXtBottleneck
from config import TEST_DATASETS, BAND_SPACING, MODEL_PATH, LOGS_PATH, MODEL_PATH, NORMALIZATION_FACTOR, NUMBER_OF_BANDS, checkpoint_fileprestring, classification_checkpoint_fileprestring, checkpoint_file, mobile_model_file, var_name, onnx_file_name, tf_model_dir, tflite_filename

import matplotlib.pyplot as plt

class AverageMeter(object):
	"""Computes and stores the average and current value."""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum/self.count

def average(list):
	return sum(list)/len(list)

activations = {}
def get_activation(name):
	def hook(model, input, output):
		activations[name] = output.detach()
	return hook

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

def read_image(rgb_filename, nir_filename):
	""" Reads the two images and stack them together while maintaining the order BGR-NIR """
	rgb = imread(rgb_filename)
	rgb[:,:, [0, 2]] = rgb[:,:, [2, 0]]	# flipping red and blue channels (shape used for training)

	nir = imread(nir_filename)
	# because NIR from the phone is saved as three repeated channels
	nir = nir[:,:, 0] if nir.ndim == 3 else np.expand_dims(nir, axis=-1)

	image = np.dstack((rgb, nir))/255.0
	del rgb, nir

	return image

def crop_image(image, start, end):
	""" Crops the image to the desired range. 
		Note: This function expects the image to be in the format [C, H, W] and H = W. """
	return image[:, start:end, start:end]

def scale_image(image, range=(0, 1)):
	""" Scales the image to the desired range.
		Depreciated: Will be removed in the future. """
	dist = image.max(dim=1, keepdim=True)[0] - image.min(dim=1, keepdim=True)[0]
	dist[dist == 0.] = 1.
	scale = 1.0 / dist
	image.mul_(scale).sub_(image.min(dim=1, keepdim=True)[0])
	image.mul_(range[1] - range[0]).add_(range[0])
	return image

def visualize_data_item(image, hypercube, band, classlabel):
	fig, ax = plt.subplots(1, 2, figsize=(10, 5))
	fig.suptitle("Class: %s" % TEST_DATASETS[classlabel])

	# visualizing it in RGB (instead of BGR)
	image=np.transpose(image.numpy()[:3], (1, 2, 0))[:,:, [2, 1, 0]]
	ax[0].imshow(image)
	ax[0].set_xlabel(image.shape)
	ax[0].set_title("RGBN - 0:3 (RGB)")
	ax[1].imshow(hypercube.numpy()[band])
	ax[1].set_xlabel(hypercube.numpy().shape)
	ax[1].set_title("Hypercube - %i" % band)
	plt.show()

def get_normalization_parameters(dataloader):
	""" Give Dataloader and recieve the mean and std of the dataset.
		Note: Make sure that the dataloader is Tensordataset and its not already normalized. """
	image_channels_sum, image_channels_squared_sum = 0, 0
	hypercube_channels_sum, hypercube_channels_squared_sum, num_batches = 0, 0, 0

	for image, hypercube, _, _ in dataloader:
		# Mean over batch, height and width, but not over the channels
		image_channels_sum += torch.mean(image, dim=[0, 2, 3])
		image_channels_squared_sum += torch.mean(image**2, dim=[0, 2, 3])

		hypercube_channels_sum += torch.mean(hypercube, dim=[0, 2, 3])
		hypercube_channels_squared_sum += torch.mean(hypercube**2, dim=[0, 2, 3])

		num_batches += 1

	image_mean = image_channels_sum / num_batches
	hypercube_mean = hypercube_channels_sum / num_batches

	# std = sqrt(E[X^2] - (E[X])^2)
	image_std = (image_channels_squared_sum / num_batches - image_mean ** 2) ** 0.5
	hypercube_std = (hypercube_channels_squared_sum / num_batches - hypercube_mean ** 2) ** 0.5

	return (image_mean, image_std), (hypercube_mean, hypercube_std)

def data_augmentation(image, aug_mode=0):
	if aug_mode == 0:
		return image								# original image
	elif aug_mode == 1:
		return np.flipud(image)						# flip up and down
	elif aug_mode == 2:
		return np.rot90(image)						# rotate counterwise 90 degree
	elif aug_mode == 3:
		return np.flipud(np.rot90(image))			# rotate 90 degree and flip up and down
	elif aug_mode == 4:
		return np.rot90(image, k=2)					# rotate 180 degree
	elif aug_mode == 5:
		return np.flipud(np.rot90(image, k=2))		# rotate 180 degree and flip
	elif aug_mode == 6:
		return np.rot90(image, k=3)					# rotate 270 degree
	elif aug_mode == 7:
		return np.flipud(np.rot90(image, k=3))		# rotate 270 degree and flip
	else:
		return

def initialize_logger(filename):
	"""Print the results in the log file."""
	logger = logging.getLogger()
	fhandler = logging.FileHandler(filename=os.path.join(LOGS_PATH, filename), mode="a")
	formatter = logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")
	fhandler.setFormatter(formatter)
	logger.addHandler(fhandler)
	logger.setLevel(logging.INFO)
	return logger

def save_checkpoint(epoch, iteration, model, optimizer, val_loss, val_acc, task="reconstruction"):
	"""Save the checkpoint."""
	state = {"epoch": epoch,
			 "iter": iteration,
			 "state_dict": model.state_dict(),
			 "optimizer": optimizer.state_dict(),
			 "val_loss": val_loss,
			 "val_acc": val_acc}

	torch.save(state, os.path.join(MODEL_PATH, task, "MS_%s_%s.pkl" % (checkpoint_fileprestring if task=="reconstruction" else classification_checkpoint_fileprestring, str(epoch).zfill(3))))

def get_best_checkpoint(task="reconstruction"):
	"""Get the model with best validation loss and validation accuracy."""
	global MODEL_PATH
	best_val_loss, best_val_acc = 0, 0
	best_checkpoint_file = None
	print("\nLoading the best checkpoint...", end="\n\n")

	for checkpoint_file in sorted(glob(os.path.join(MODEL_PATH, task, "*.pkl"))):
		if os.path.isfile(checkpoint_file):
			save_point = torch.load(checkpoint_file)
			val_loss = save_point["val_loss"]
			val_acc = save_point["val_acc"]
			print("Checkpoint: %s\tValidation Loss: %.9f\tValidation Accuracy: %.2f%%" % (checkpoint_file.split("/")[-1], val_loss, val_acc), end="\t")
			if (100 - best_val_loss + best_val_acc) < (100 - val_loss + val_acc):
				print("<- Best checkpoint yet. Updating the best checkpoint.", end="")
				best_val_loss = val_loss
				best_val_acc = val_acc
				best_checkpoint_file = checkpoint_file.split("/")[-1]
			print()
	print("\nThe best checkpoint file, is loaded, for %s task and it is %s with validation loss value %.9f and validation accuracy %.2f%%" %
		  (task, best_checkpoint_file, best_val_loss, best_val_acc), end="\n\n")

	loaded_model = torch.load(os.path.join(MODEL_PATH, task, best_checkpoint_file))
	return loaded_model["epoch"], loaded_model["iter"], loaded_model["state_dict"], loaded_model["optimizer"], loaded_model["val_loss"], loaded_model["val_acc"]

def save_matv73(mat_filename, hypercube):
	hdf5storage.savemat(mat_filename, {var_name: hypercube}, format="7.3", store_python_metadata=True)

class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
	""" Record many results."""
	loss_csv.write("{}, {}, {}, {}, {}, {}\n".format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
	loss_csv.flush()
	loss_csv.close

def get_reconstruction(input, num_split, dimension, model):
	"""As the limited GPU memory split the input."""
	input_split = torch.split(input, int(input.shape[3]/num_split), dim=dimension)
	output_split = []
	for i in range(num_split):
		with torch.no_grad():
			var_input = Variable(input_split[i].cuda())
		var_output = model(var_input)
		output_split.append(var_output.data)
		if i == 0:
			output = output_split[i]
		else:
			output = torch.cat((output, output_split[i]), dim=dimension)
	return output

def reconstruction(rgb, model, normalize=False):
	"""Output the final reconstructed hyperspectral images."""
	img_res = get_reconstruction(rgb.float(), 1, 3, model)
	img_res = img_res.cpu().numpy()
	img_res = np.transpose(np.squeeze(img_res))
	if normalize:
		img_res = img_res / NORMALIZATION_FACTOR
		img_res = np.minimum(img_res, NORMALIZATION_FACTOR)
		img_res = np.maximum(img_res, 0)
	return img_res

def load_mat(mat_name):
	""" Helper function to load mat files (used in making h5 dataset) """
	data = hdf5storage.loadmat(mat_name, variable_names=[var_name])
	return data[var_name] / NORMALIZATION_FACTOR

def make_h5_dataset(DATASET_DIR, h5_filename):
	labels = []
	images = []

	for filename in glob(os.path.join(DATASET_DIR, "*_dense_demRGB.png")):
		mat_file_name = filename.split("/")[-1].split("_")[0]
		rgb_img_path = filename
		nir_img_path = os.path.join(DATASET_DIR, filename.split("/")[-1].replace("RGB", "NIRc"))
		print(rgb_img_path)
		rgb = imread(rgb_img_path)
		rgb[:,:, [0, 2]] = rgb[:,:, [2, 0]]	# flipping red and blue channels (shape used for training)

		nir = imread(nir_img_path)
		# because NIR from the phone is saved as three repeated channels
		nir = nir[:,:, 0] if nir.ndim == 3 else np.expand_dims(nir, axis=-1)

		image = np.dstack((rgb, nir))/255.0
		image = np.transpose(image, [2, 0, 1])

		ground_t = load_mat(os.path.join(os.path.dirname(DATASET_DIR), "mat", mat_file_name + ".mat"))
		ground_t = ground_t[:, :, ::BAND_SPACING]
		ground_t = np.transpose(ground_t, [2, 0, 1])

		images.append(image)
		labels.append(ground_t)

	print("Creating h5 file at %s" % os.path.join(os.path.dirname(DATASET_DIR), h5_filename))
	hf = h5py.File(os.path.join(os.path.dirname(DATASET_DIR), h5_filename), "w")

	hf = h5py.File(os.path.join("datasets", h5_filename), "w")
	hf.create_dataset("data", dtype=np.float32, data=images)
	hf.create_dataset("label", dtype=np.float32, data=labels)
	hf.close()

def makeMobileModel():
	save_point = torch.load(os.path.join(MODEL_PATH, checkpoint_file))
	model_param = save_point["state_dict"]
	model = Network(block=ResNeXtBottleneck, block_num=10, input_channel=4, n_hidden=64, output_channel=NUMBER_OF_BANDS)
	model.load_state_dict(model_param)

	model.eval()
	input_tensor = torch.rand(1, 4, 512, 512)

	script_model = torch.jit.trace(model, input_tensor)
	script_model_optimized = optimize_for_mobile(script_model)
	script_model_optimized.save(os.path.join(LOGS_PATH, mobile_model_file))

def modeltoONNX():
	save_point = torch.load(os.path.join(MODEL_PATH, checkpoint_file))
	model_param = save_point["state_dict"]
	model = Network(block=ResNeXtBottleneck, block_num=10, input_channel=4, n_hidden=64, output_channel=NUMBER_OF_BANDS)
	model.load_state_dict(model_param)

	model.eval()
	input_tensor = torch.rand(1, 4, 512, 512)

	torch.onnx.export(model, input_tensor, os.path.join(LOGS_PATH, onnx_file_name), export_params=True, verbose=True)

def ONNXtotf():
	import onnx
	import onnx_tf

	model = onnx.load(os.path.join(LOGS_PATH, onnx_file_name))
	tf_model = onnx_tf.backend.prepare(model)
	tf_model.export_graph(tf_model_dir)

def tf_to_tflite():
	import tensorflow as tf

	converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)		# path to the SavedModel directory
	converter.target_spec.supported_ops = [
		tf.lite.OpsSet.TFLITE_BUILTINS,		# enable TFLite ops
		tf.lite.OpsSet.SELECT_TF_OPS		# enable TF ops
	]
	tflite_model = converter.convert()

	# Save the model
	with open(os.path.join(LOGS_PATH, tflite_filename), "wb") as f:
		f.write(tflite_model)