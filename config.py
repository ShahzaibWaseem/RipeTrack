import os

TRAIN_DATASET_DIR = os.path.join("..", "data_preparation", "datasets")

TRAIN_DATASET_FILES = ["train_avocado_halogen_4to51bands.h5",
					   "train_apple_halogen_4to51bands.h5",
					   "train_apple_cfl_led_4to51bands.h5",
					   "train_avocado_cfl_led_4to51bands.h5"]
VALID_DATASET_FILES = ["valid_avocado_halogen_4to51bands.h5",
					   "valid_apple_halogen_4to51bands.h5",
					   "valid_apple_cfl_led_4to51bands.h5",
					   "valid_avocado_cfl_led_4to51bands.h5"]
batch_size = 64
end_epoch = 51
init_lr = 0.0001
fusion_techniques = ["add", "concat"]

MODEL_NAME = "resnext"
ILLUMINATIONS = ["h", "cfl_led"]

if "h" in ILLUMINATIONS and "cfl_led" in ILLUMINATIONS:
	illumination_string = "halogen + CFL-LED"
elif "h" in ILLUMINATIONS:
	illumination_string = "halogen"
elif "cfl_led" in ILLUMINATIONS:
	illumination_string = "CFL-LED"

model_run_title = "Model: %s\tDataset: %s\tIllumination: %s\tLosses: SAM + MRAE\tFull Image or Patches: patches\n" % (MODEL_NAME, DATASET_NAME, illumination_string)

MODEL_PATH = os.path.join(".", "checkpoints")
LOGS_PATH = os.path.join(".", "logs")

TEST_ROOT_DATASET_DIR = os.path.join(os.path.dirname(TRAIN_DATASET_DIR), "working_datasets")
TEST_DATASETS = ["avocado", "apple"]
ILLUMINATIONS = ["h", "cfl_led"]

checkpoint_fileprestring = "%s_%s" % (MODEL_NAME, DATASET_NAME)
checkpoint_file = "MS_%s_23.pkl" % checkpoint_fileprestring
mobile_model_file = "model_%s.pth" % DATASET_NAME
onnx_file_name = "model.onnx"
tf_model_dir = os.path.join("tfmodel")
tflite_filename = "model.tflite"
# checkpoint_file = "HS_model_%d.pkl" % end_epoch

plt_dict = {"mathtext.default": "regular",
			"axes.linewidth": 2}

text_font_dict = {"family": "serif",
				  "size": 25}

title_font_dict = {"fontname": "serif",
				   "size": 25}

VIEW_BANDS = [11, 21, 36, 44]
ACTUAL_BANDS = [520, 640, 820, 910]

var_name = "rad"

def init_directories():
	for directory in [MODEL_PATH, LOGS_PATH]:
		if not os.path.exists(directory):
			os.makedirs(directory)

	for fusion in fusion_techniques:
		if not os.path.exists(os.path.join(MODEL_PATH, fusion)):
			os.makedirs(os.path.join(MODEL_PATH, fusion))

	for test_dataset in TEST_DATASETS:
		for illumination in ILLUMINATIONS:
			for directory in ["inference", "images"]:
				test_dataset_path = os.path.join(TEST_ROOT_DATASET_DIR, "working_%s" % test_dataset, "%s_%s_204ch" % (test_dataset, illumination), "test", directory, MODEL_NAME)
					if not os.path.exists(test_dataset_path):
						os.makedirs(test_dataset_path)