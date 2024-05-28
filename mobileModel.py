import os
import pickle

import torch
import sklearn
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from torch.utils.mobile_optimizer import optimize_for_mobile

from models.MST import MST_Plus_Plus
from models.classifier import ModelWithAttention

from utils import create_directory, get_best_checkpoint
from config import MODEL_PATH, MOBILE_MODELS_DIR_NAME, LABELS_DICT, TIME_LEFT_DICT, BANDS

def makeMobileModel(checkpoint, task="reconstruction", torch_mobile_model_filename="RipeTrack_reconstruction_mobile_68.pt"):
	create_directory(os.path.join(MODEL_PATH, MOBILE_MODELS_DIR_NAME))
	# best_checkpoint_file, epoch, iter, state_dict, opt_state, val_loss, val_acc = get_best_checkpoint(task="classification")
	
	state_dict = checkpoint["state_dict"]
	model = MST_Plus_Plus(in_channels=4, out_channels=len(BANDS), n_feat=len(BANDS)//2, msab_stages=2, stage=1) if task == "reconstruction" \
		else ModelWithAttention(input_channels=len(BANDS), num_classes=len(LABELS_DICT), num_subclasses=len(TIME_LEFT_DICT))
	model.load_state_dict(state_dict)
	model.eval()
	input_tensor = torch.rand(1, 4, 64, 64) if task == "reconstruction" else torch.rand(1, len(BANDS), 64, 64)

	model = torch.quantization.convert(model)

	script_model = torch.jit.trace(model, input_tensor)
	script_model_optimized = optimize_for_mobile(script_model)
	script_model_optimized._save_for_lite_interpreter(os.path.join(MODEL_PATH, MOBILE_MODELS_DIR_NAME, torch_mobile_model_filename))

def sklernModelToONNX(sklearn_model_filename="MLP_slp_k0.pkl", onnx_model_filename="MLP_slp_k0.onnx"):
	pipeline = pickle.load(open(os.path.join(MODEL_PATH, MOBILE_MODELS_DIR_NAME, sklearn_model_filename), "rb"))
	print(pipeline)

	initial_type = [("float_input", FloatTensorType([None, len(BANDS)]))]

	onnx = convert_sklearn(pipeline, initial_types=initial_type)
	with open(os.path.join(MODEL_PATH, MOBILE_MODELS_DIR_NAME, onnx_model_filename), "wb") as file:
		file.write(onnx.SerializeToString())

if __name__ == "__main__":
	print(torch.__version__)
	checkpoint_filename = "RT_MST++_shelflife_080 RGBNIR Final [ThinModel][L+A].pkl"
	# checkpoint_filename = "MSLP_ModelWithAttention_shelflife_100.pkl"
	checkpoint = torch.load(os.path.join(MODEL_PATH, "reconstruction", "pre-trained", checkpoint_filename))
	makeMobileModel(checkpoint, task="reconstruction")
	# sklernModelToONNX()