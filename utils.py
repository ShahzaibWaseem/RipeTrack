from __future__ import division

import os
import numpy as np

import h5py
import hdf5storage

import logging
from glob import glob
from imageio import imread

import tensorflow as tf

import onnx
import onnx_tf

import torch
from torch.autograd import Variable
from torch.utils.mobile_optimizer import optimize_for_mobile

from models.model import Network
from models.resblock import resblock, ResNeXtBottleneck

from config import MODEL_PATH, LOGS_PATH, MODEL_PATH, checkpoint_fileprestring, checkpoint_file, mobile_model_file, var_name, onnx_file_name, tf_model_dir, tflite_filename

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

def save_checkpoint(epoch, iteration, model, optimizer):
	"""Save the checkpoint."""
	state = {"epoch": epoch,
			 "iter": iteration,
			 "state_dict": model.state_dict(),
			 "optimizer": optimizer.state_dict()}
	torch.save(state, os.path.join(MODEL_PATH, "MS_%s_%d.pkl" % (checkpoint_fileprestring, epoch)))

def save_matv73(mat_name, var_name, var):
	hdf5storage.savemat(mat_name, {var_name: var}, format="7.3", store_python_metadata=True)

def record_loss(loss_csv,epoch, iteration, epoch_time, lr, train_loss, test_loss):
	""" Record many results."""
	loss_csv.write("{},{},{},{},{},{}\n".format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
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

def reconstruction(rgb, model):
	"""Output the final reconstructed hyperspectral images."""
	img_res = get_reconstruction(torch.from_numpy(rgb).float(), 1, 3, model)
	img_res = img_res.cpu().numpy()*4095
	img_res = np.transpose(np.squeeze(img_res))
	img_res_limits = np.minimum(img_res,4095)
	img_res_limits = np.maximum(img_res_limits,0)
	return img_res_limits

def load_mat(mat_name, var_name):
	""" Helper function to load mat files (used in making h5 dataset) """
	data = hdf5storage.loadmat(mat_name, variable_names=[var_name])
	return data

def make_h5_dataset(DATASET_DIR, h5_filename):
	labels = []
	images = []

	for filename in glob(os.path.join(DATASET_DIR, "*_dense_demRGB.png")):
		mat_file_name = filename.split("/")[-1].split("_")[0]
		rgb_img_path = filename
		nir_img_path = os.path.join(DATASET_DIR, filename.split("/")[-1].replace("RGB", "NIRc"))
		print(rgb_img_path)
		rgb = imread(rgb_img_path)
		rgb = rgb/255
		rgb[:,:, [0, 2]] = rgb[:,:, [2, 0]]		# flipping red and blue channels (shape used for training)
		
		nir = imread(nir_img_path)/255
		image = np.dstack((rgb, nir))
		image = np.transpose(image, [2, 0, 1])

		ground_t = load_mat(os.path.join(os.path.dirname(DATASET_DIR), "mat", mat_file_name + ".mat"), var_name)
		ground_t = ground_t[var_name][:, :, 1:204:4]/4095
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
	model = Network(block=ResNeXtBottleneck, block_num=10, input_channel=4, n_hidden=64, output_channel=51)
	model.load_state_dict(model_param)

	model.eval()
	input_tensor = torch.rand(1, 4, 512, 512)

	script_model = torch.jit.trace(model, input_tensor)
	script_model_optimized = optimize_for_mobile(script_model)
	script_model_optimized.save(os.path.join(LOGS_PATH, mobile_model_file))

def modeltoONNX():
	save_point = torch.load(os.path.join(MODEL_PATH, checkpoint_file))
	model_param = save_point["state_dict"]
	model = Network(block=ResNeXtBottleneck, block_num=10, input_channel=4, n_hidden=64, output_channel=51)
	model.load_state_dict(model_param)

	model.eval()
	input_tensor = torch.rand(1, 4, 512, 512)

	torch.onnx.export(model, input_tensor, os.path.join(LOGS_PATH, onnx_file_name), export_params=True, verbose=True)

def ONNXtotf():
	model = onnx.load(os.path.join(LOGS_PATH, onnx_file_name))
	tf_model = onnx_tf.backend.prepare(model)
	tf_model.export_graph(tf_model_dir)

def tf_to_tflite():
	converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)		# path to the SavedModel directory
	converter.target_spec.supported_ops = [
		tf.lite.OpsSet.TFLITE_BUILTINS,		# enable TFLite ops
		tf.lite.OpsSet.SELECT_TF_OPS		# enable TF ops
	]
	tflite_model = converter.convert()

	# Save the model
	with open(os.path.join(LOGS_PATH, tflite_filename), "wb") as f:
		f.write(tflite_model)