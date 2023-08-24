import os

var_name = "hcube"					# key for the dictionary which are saved in the files

SHELF_LIFE_GROUND_TRUTH_FILENAME = "ShelfLifeGroundTruth.csv"

### Datagenerator Directories ###
CAMERA_OUTPUT_ROOT_PATH = os.path.join("..", "Catalog")
CAMERA_EXTRACT_DATASETS = ["pear-bosc", "pear-williams", "avocado-organic", "avocado-emp"]

### Root directories for train and test dataset loading ###
TRAIN_DATASET_DIR = os.path.join("..", "data_preparation", "h5datasets")

TEST_ROOT_DATASET_DIR = os.path.join(os.path.dirname(TRAIN_DATASET_DIR), "datasets")
TEST_DATASETS = ["pear-bosc", "pear-williams", "avocado-organic", "avocado-emp"]
LABELS = {"SlightlyUnripe": 0, "Ripe": 1, "Dangerous": 2, "Expired": 3}

### used only in train.py (these h5 files contain patches of the datasets) ###
# TRAIN_DATASET_FILES = ["train_avocado_halogen_4to51bands.h5",
# 					   "train_apple_halogen_4to51bands.h5",
# 					   "train_apple_cfl_led_4to51bands.h5",
# 					   "train_avocado_cfl_led_4to51bands.h5"]
# VALID_DATASET_FILES = ["valid_avocado_halogen_4to51bands.h5",
# 					   "valid_apple_halogen_4to51bands.h5",
# 					   "valid_apple_cfl_led_4to51bands.h5",
# 					   "valid_avocado_cfl_led_4to51bands.h5"]

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

### Hyperparamters for the model ###
batch_size = 32
end_epoch = 501
init_lr = 0.0001

### Variables used for printing the results in the logs ###
APPLICATION_NAME = "shelflife"
MODEL_NAME = "resnext"
CLASSIFIER_MODEL_NAME = "efficientnet-b4"
ILLUMINATIONS = ["h"]

if "h" in ILLUMINATIONS and "cfl_led" in ILLUMINATIONS:
	illumination_string = "halogen + CFL-LED"
elif "h" in ILLUMINATIONS:
	illumination_string = "halogen"
elif "cfl_led" in ILLUMINATIONS:
	illumination_string = "CFL-LED"

model_run_title = "Model: %s\tDataset: %s\tIllumination: %s\tLosses: MRAE + SAM + SID + Weighted\tFull Image or Patches: %s\n" \
	% (MODEL_NAME, APPLICATION_NAME, illumination_string, "Full Image" if PATCH_SIZE == IMAGE_SIZE else "Patches")
classicication_run_title = "Model: %s\tDataset: %s\tIllumination: %s\tNumber of Classes: %d\tFull Image or Patches: %s\n" \
	% (CLASSIFIER_MODEL_NAME, APPLICATION_NAME, illumination_string, len(TEST_DATASETS), "Full Image" if PATCH_SIZE == IMAGE_SIZE else "Patches")

### to create the checkpoint of the model ###
checkpoint_fileprestring = "%s_%s" % (MODEL_NAME, APPLICATION_NAME)
classification_checkpoint_fileprestring = "%s_%s" % (CLASSIFIER_MODEL_NAME, APPLICATION_NAME)
checkpoint_file = "MS_%s_500.pkl" % checkpoint_fileprestring
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

### Predefined Transforms (Just to speed the process up)
from torchvision import transforms

predef_input_transform = transforms.Compose([transforms.Normalize(
	mean=[0.0752, 0.1171, 0.2212, 0.1676],
	std=[0.0726, 0.1514, 0.2097, 0.1509])])

predef_label_transform = transforms.Compose([transforms.Normalize(
	mean=[0.0049, 0.0068, 0.0167, 0.0215, 0.0218, 0.0225, 0.0229, 0.0231, 0.0232,
	0.0232, 0.0231, 0.0232, 0.0233, 0.0234, 0.0237, 0.0237, 0.0237, 0.0240,
	0.0241, 0.0242, 0.0244, 0.0245, 0.0246, 0.0246, 0.0245, 0.0244, 0.0241,
	0.0240, 0.0239, 0.0239, 0.0238, 0.0237, 0.0237, 0.0237, 0.0235, 0.0234,
	0.0233, 0.0232, 0.0231, 0.0230, 0.0228, 0.0225, 0.0222, 0.0217, 0.0213,
	0.0207, 0.0201, 0.0199, 0.0200, 0.0201, 0.0204, 0.0207, 0.0212, 0.0215],
	std=[0.0045, 0.0088, 0.0160, 0.0189, 0.0221, 0.0229, 0.0233, 0.0235, 0.0235,
	0.0233, 0.0231, 0.0231, 0.0231, 0.0233, 0.0236, 0.0235, 0.0235, 0.0237,
	0.0238, 0.0240, 0.0242, 0.0243, 0.0244, 0.0245, 0.0244, 0.0241, 0.0238,
	0.0236, 0.0235, 0.0234, 0.0233, 0.0232, 0.0231, 0.0230, 0.0228, 0.0226,
	0.0224, 0.0222, 0.0220, 0.0217, 0.0213, 0.0208, 0.0202, 0.0193, 0.0185,
	0.0174, 0.0165, 0.0160, 0.0158, 0.0156, 0.0157, 0.0157, 0.0159, 0.0160])])

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

def init_directories():
	""" Creates directories if they don't exist which are used to save model, logs, generated hypercubes and visualizations """
	for directory in [MODEL_PATH, LOGS_PATH]:
		if directory == MODEL_PATH:
			create_directory(os.path.join(directory, "reconstruction"))
			create_directory(os.path.join(directory, "classification"))
		create_directory(directory)

	for test_dataset in TEST_DATASETS:
		for illumination in ILLUMINATIONS:
			for directory in ["inference", "images"]:
				# ignore making directories for chicken cfl-led dataset
				if illumination == "cfl_led" and test_dataset == "chicken":
					continue
				test_dataset_path = os.path.join(TEST_ROOT_DATASET_DIR, "working_%s" % APPLICATION_NAME, "working_%s" % test_dataset, "%s_%s_204ch" \
					% (test_dataset, illumination), "test", directory, MODEL_NAME)
				create_directory(test_dataset_path)