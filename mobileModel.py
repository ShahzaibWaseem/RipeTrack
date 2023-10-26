import os
import pickle

import torch
import sklearn
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from torch.utils.mobile_optimizer import optimize_for_mobile

from utils import get_best_checkpoint
from models.MST import MST_Plus_Plus
from config import MODEL_PATH, BANDS, MOBILE_MODELS_DIR_NAME

def makeMobileModel(torch_mobile_model_filename="mobile_mst_68.pt"):
	best_checkpoint_file, epoch, iter, state_dict, opt_state, val_loss, val_acc = get_best_checkpoint(task="reconstruction")
	model = MST_Plus_Plus(in_channels=4, out_channels=len(BANDS), n_feat=len(BANDS), stage=3)
	model.load_state_dict(state_dict)
	model.eval()
	input_tensor = torch.rand(1, 4, 512, 512)

	model = torch.quantization.convert(model)

	script_model = torch.jit.trace(model, input_tensor)
	script_model_optimized = optimize_for_mobile(script_model)
	script_model_optimized._save_for_lite_interpreter(os.path.join(MODEL_PATH, MOBILE_MODELS_DIR_NAME, torch_mobile_model_filename))

def sklernModelToONNX(sklearn_model_filename="MLP_slp_k0.pkl", onnx_model_filename="MLP_slp_k0.onnx"):
	pipeline = pickle.load(open(os.path.join(MODEL_PATH, MOBILE_MODELS_DIR_NAME, sklearn_model_filename), "rb"))
	print(pipeline)

	initial_type = [("float_input", FloatTensorType([None, 68]))]

	onnx = convert_sklearn(pipeline, initial_types=initial_type)
	with open(os.path.join(MODEL_PATH, MOBILE_MODELS_DIR_NAME, onnx_model_filename), "wb") as file:
		file.write(onnx.SerializeToString())

if __name__ == "__main__":
	print(sklearn.__version__)
	makeMobileModel()
	# sklernModelToONNX()