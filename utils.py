from __future__ import division

import os
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
from models.resblock import resblock, ResNeXtBottleneck

from config import BAND_SPACING, MODEL_PATH, LOGS_PATH, MODEL_PATH, NORMALIZATION_FACTOR, NUMBER_OF_BANDS, checkpoint_fileprestring, classification_checkpoint_fileprestring, checkpoint_file, mobile_model_file, var_name, onnx_file_name, tf_model_dir, tflite_filename

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

	torch.save(state, os.path.join(MODEL_PATH, task, "MS_%s_%d.pkl" % (checkpoint_fileprestring if task=="reconstruction" else classification_checkpoint_fileprestring, epoch)))

def get_best_checkpoint(task="reconstruction", up_a_directory=False):
	"""Get the model with best validation loss and validation accuracy."""
	global MODEL_PATH
	bes_val_loss, best_val_acc = 0.0, 0.0
	best_checkpoint_file = None
	MODEL_PATH = os.path.join("..", MODEL_PATH) if up_a_directory else MODEL_PATH
	print("Loading the best checkpoint...")

	for checkpoint_file in glob(os.path.join(MODEL_PATH, task, "*.pkl")):
		if os.path.isfile(checkpoint_file):
			save_point = torch.load(checkpoint_file)
			val_loss = save_point["val_loss"]
			val_acc = save_point["val_acc"]
			print("Checkpoint: {}\tValidation Loss: {}, Validation Accuracy: {}".format(checkpoint_file, val_loss, val_acc))
			if (100 - val_loss + val_acc) < (100 - bes_val_loss + best_val_acc):
				bes_val_loss = val_loss
				best_val_acc = val_acc
				best_checkpoint_file = checkpoint_file
	print("The best checkpoint file, is loaded, for task %s is %s with validation loss value %.9f and validation accuracy %.2f%%" %
		  (task, best_checkpoint_file, bes_val_loss, best_val_acc))

	loaded_model = torch.load(os.path.join(MODEL_PATH, task, best_checkpoint_file))
	return loaded_model["epoch"], loaded_model["iter"], loaded_model["state_dict"], loaded_model["optimizer"], loaded_model["val_loss"], loaded_model["val_acc"]

def save_matv73(mat_filename, hypercube):
	hdf5storage.savemat(mat_filename, {var_name: hypercube}, format="7.3", store_python_metadata=True)

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