import os
import torch

var_name = "hcube"					# key for the dictionary which are saved in the files
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 0.00001

SHELF_LIFE_GROUND_TRUTH_FILENAME = "ShelfLifeGroundTruth.csv"

### Datagenerator Directories ###
CAMERA_OUTPUT_ROOT_PATH = os.path.join("..", "Catalog")
CAMERA_EXTRACT_DATASETS = ["pear-bosc", "pear-williams", "avocado-organic", "avocado-emp"]

### Root directories for train and test dataset loading ###
TRAIN_DATASET_DIR = os.path.join("..", "data_preparation", "h5datasets")
TEST_ROOT_DATASET_DIR = os.path.join(os.path.dirname(TRAIN_DATASET_DIR), "datasets")
TEST_DATASETS = ["pear-bosc", "pear-williams", "avocado-organic", "avocado-emp"]
LABELS = {"SlightlyUnripe": 0, "Ripe": 1, "Dangerous": 2, "Expired": 3}

GT_RGBN_DIR_NAME = "rgbn"
GT_SECONDARY_RGB_CAM_DIR_NAME = "secondary-rgbn"
MOBILE_DATASET_DIR_NAME = "mobile-rgbn"
RECONSTRUCTED_HS_DIR_NAME = "reconstructed"
MOBILE_RECONSTRUCTED_HS_DIR_NAME = "mobile-reconstructed"
VISUALIZATION_DIR_NAME = "visualizations"

APPEND_SECONDARY_RGB_CAM_INPUT = True

PREDEF_TRANSFORMS_FILENAME = "transforms{}.pth".format("_appended" if APPEND_SECONDARY_RGB_CAM_INPUT else "")

# used only in train.py (these h5 files contain patches of the datasets) ###
TRAIN_DATASET_FILES = ["train_avocado_halogen_4to51bands.h5",
					   "train_apple_halogen_4to51bands.h5",
					   "train_apple_cfl_led_4to51bands.h5",
					   "train_avocado_cfl_led_4to51bands.h5"]
VALID_DATASET_FILES = ["valid_avocado_halogen_4to51bands.h5",
					   "valid_apple_halogen_4to51bands.h5",
					   "valid_apple_cfl_led_4to51bands.h5",
					   "valid_avocado_cfl_led_4to51bands.h5"]

### Directories for logs and model checkpoints ###
MODEL_PATH = os.path.join(".", "checkpoints")
LOGS_PATH = os.path.join(".", "logs")

### Parameters for Data reading ###
BAND_SPACING = 1						# used only if reading data from directories dataset.DatasetFromDirectory
NUMBER_OF_BANDS = 204//BAND_SPACING		# holds the number of bands considered (used in model creation)

IMAGE_SIZE = 512
PATCH_SIZE = 64							# patching of the hypercubes and images
NORMALIZATION_FACTOR = 4096				# max value of the captured hypercube (dependent on the camera - Specim IQ)
RGBN_BANDS = [18, 47, 80, 183]			# correspond to B 454, G 541, R 640, N 949 bands
# BANDS = [RGBN_BANDS, range(104, 204, 2)]
# Bands: (450, 650, 5), (670, 690, 5), (750, 900, 5), (900, 1000, 5)
BANDS = [range(19, 87, 5), range(94, 102, 5), range(120, 170, 5), range(170, 204, 5)]
# BANDS = [range(0, 204, 3)]

### Hyperparamters for the model ###
batch_size = 16
end_epoch = 501
init_lr = 0.0001

### Variables used for printing the results in the logs ###
APPLICATION_NAME = "shelflife"
MODEL_NAME = "MST++"
CLASSIFIER_MODEL_NAME = "efficientnet-b4"
ILLUMINATIONS = ["h"]

if "h" in ILLUMINATIONS and "cfl_led" in ILLUMINATIONS:
	illumination_string = "halogen + CFL-LED"
elif "h" in ILLUMINATIONS:
	illumination_string = "halogen"
elif "cfl_led" in ILLUMINATIONS:
	illumination_string = "CFL-LED"

lossfunctions_considered = ["MRAE", "SAM", "SID"]
model_run_title = "Model: %s\tDataset: %s\tIllumination: %s\tLosses: %s\tFull Image or Patches: %s\n" \
	% (MODEL_NAME, APPLICATION_NAME, illumination_string, lossfunctions_considered, "Full Image" if PATCH_SIZE == IMAGE_SIZE else "Patches")
classicication_run_title = "Model: %s\tDataset: %s\tIllumination: %s\tNumber of Classes: %d\tFull Image or Patches: %s\n" \
	% (CLASSIFIER_MODEL_NAME, APPLICATION_NAME, illumination_string, len(TEST_DATASETS), "Full Image" if PATCH_SIZE == IMAGE_SIZE else "Patches")

### to create the checkpoint of the model ###
checkpoint_fileprestring = "%s_%s" % (MODEL_NAME, APPLICATION_NAME)
classification_checkpoint_fileprestring = "%s_%s" % (CLASSIFIER_MODEL_NAME, APPLICATION_NAME)
checkpoint_file = "MSLP_%s_499.pkl" % checkpoint_fileprestring
# checkpoint_file = "HS_model_%d.pkl" % end_epoch
run_pretrained = False					# if True, the model is loaded from the checkpoint_file

mobile_model_file = "model_%s.pth" % APPLICATION_NAME
onnx_file_name = "model.onnx"
tf_model_dir = os.path.join("tfmodel")
tflite_filename = "model.tflite"

### Formatting used for the visualizations ###
plt_dict = {"mathtext.default": "regular", "axes.linewidth": 2}
text_font_dict = {"family": "serif", "size": 25}
title_font_dict = {"fontname": "serif", "size": 25}

### Bands for text in the visualizations ###
VIEW_BANDS = [11, 21, 36, 44]
ACTUAL_BANDS = [520, 640, 820, 910]

def sampler():
	global BANDS, NUMBER_OF_BANDS
	sampled_bands = []

	for band_segment in range(len(BANDS)):
		sampled_bands.append(list(BANDS[band_segment]))
	
	BANDS = [band for sampled_list in sampled_bands for band in sampled_list]
	NUMBER_OF_BANDS = len(BANDS)
	return BANDS

sampler()

def create_directory(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)