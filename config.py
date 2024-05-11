import os
import torch
from collections import OrderedDict

var_name = "hcube"					# key for the dictionary which are saved in the files
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 0.0001

GT_DATASET_CROPS_FILENAME = "gtShelfLifeCrops.csv"
MOBILE_DATASET_CROPS_FILENAME = "mobileShelfLifeCrops.csv"
MOBILE_OP_DATASET_CROPS_FILENAME = "mobileOPShelfLifeCrops.csv"
SHELF_LIFE_GROUND_TRUTH_FILENAME = "ShelfLifeGroundTruth.csv"

### Datagenerator Directories ###
CAMERA_OUTPUT_ROOT_PATH = os.path.join("..", "Catalog")
CAMERA_EXTRACT_DATASETS = ["pear-bosc", "pear-bartlett", "pear-bartlettnew", "avocado-organic", "avocado-organicnew", "avocado-hass", "banana-ecuador", "banana-guatamala"]

### Root directories for train and test dataset loading ###
TRAIN_DATASET_DIR = os.path.join("..", "h5datasets")
TEST_ROOT_DATASET_DIR = os.path.join(os.path.dirname(TRAIN_DATASET_DIR), "datasets")

### Directories for logs and model checkpoints ###
MODEL_PATH = os.path.join(".", "checkpoints")
LOGS_PATH = os.path.join(".", "logs")
DATA_PREP_PATH = os.path.join(".", "dataPreparation")

### Using the following datasets for training and testing ###
TEST_DATASETS = ["pear-bosc", "pear-bartlett", "avocado-organic", "avocado-hass"]
# TEST_DATASETS = ["mango-ataulfo", "banana-ecuador", "banana-guatamala", "nectarine-chile"]

### Dictionaries for the Classification labels ###
FRUITS_DICT = OrderedDict([("Pear Bosc", 0), ("Pear Bartlett", 1), ("Avocado Organic", 2), ("Avocado Hass", 3)])
LABELS_DICT = OrderedDict([("Unripe", 0), ("Ripe", 1), ("Expired", 2)])
TIME_LEFT_DICT = OrderedDict([("100%", 0), ("90%", 1), ("80%", 2), ("70%", 3), ("60%", 4), ("50%", 5), ("40%", 6), ("30%", 7), ("20%", 8), ("10%", 9), ("0%", 10)])

### Directories for the datasets ###
GT_HYPERCUBES_DIR_NAME = "gt-cubes"
GT_RGBN_DIR_NAME = "rgbn"
GT_AUXILIARY_RGB_CAM_DIR_NAME = "secondary-rgbn"
MOBILE_DATASET_DIR_NAME = "mobile-rgbn"
OP_MOBILE_DATASET_DIR_NAME = "oneplus-rgbn"
GT_REMOVED_IR_CUTOFF_DIR_NAME = "rgbnir-sensor"
GT_REMOVED_IR_CUTOFF_RECONSTRUCTED_DIR_NAME = "rgbnir-sensor-reconstructed"
RECONSTRUCTED_HS_DIR_NAME = "reconstructed"
MOBILE_RECONSTRUCTED_HS_DIR_NAME = "mobile-reconstructed"
OP_MOBILE_RECONSTRUCTED_HS_DIR_NAME = "op-mobile-reconstructed"
RAW_MOBILE_RECONSTRUCTED_HS_DIR_NAME = "raw-mobile-reconstructed"
DISTANCE_DIR_NAME = "distance"
VISUALIZATION_DIR_NAME = "visualizations"
MOBILE_MODELS_DIR_NAME = "mobileModels"
TRAIN_VAL_TEST_SPLIT_DIR_NAME = "split"
DEEP_WB_DIR = "deepWB"

### Parameters for the models and data loading ###
APPEND_SECONDARY_RGB_CAM_INPUT = False
run_pretrained = False					# if True, the model is loaded from the checkpoint_file
use_mobile_dataset = False				# if True, the model is trained on the mobile dataset
transfer_learning = False				# if True, the model will freeze all layers except the last MST block and conv layers
# run_pretrained = True if transfer_learning else run_pretrained

PREDEF_TRANSFORMS_FILENAME = "transforms{}.pth".format("_appended" if APPEND_SECONDARY_RGB_CAM_INPUT else "")

# used only in train.py (these h5 files contain patches of the datasets) ###
TRAIN_DATASET_FILES = []
VALID_DATASET_FILES = []

### Parameters for Data reading ###
BAND_SPACING = 1						# used only if reading data from directories dataset.DatasetFromDirectory
NUMBER_OF_BANDS = 204//BAND_SPACING		# holds the number of bands considered (used in model creation)

IMAGE_SIZE = 512
PATCH_SIZE = 64							# patching of the hypercubes and images
CLASSIFICATION_PATCH_SIZE = 64			# patching of the hypercubes and images during classification
STRIDE = 5								# stride for patching the hypercubes and images
NORMALIZATION_FACTOR = 4096				# max value of the captured hypercube (dependent on the camera - Specim IQ)
RGBN_BANDS = [18, 47, 80, 183]			# correspond to B 454, G 541, R 640, N 949 bands
NIR_BANDS = list(range(183, 197))		# correspond to NIR bands range
BANDS = [range(0, 204, 3)]				# bands used in the models (used for training and testing; For mobile dataset, this is overwritten while sampling)
# actual bands numbers (used for printing and visualizations)
BANDS_WAVELENGTHS = [397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29, 426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25, 455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32, 484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48, 513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75, 542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12, 572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60, 601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18, 631.15, 634.11, 637.08, 640.04, 643.01, 645.98, 648.95, 651.92, 654.89, 657.87, 660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65, 690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54, 720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54, 750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64, 780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84, 810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14, 841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55, 871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95, 896.01, 899.06, 902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68, 932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40, 963.47, 966.55, 969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22, 994.31, 997.40, 1000.49, 1003.58]

### Hyperparamters for the model ###
batch_size = 25
end_epoch = 101
init_lr = 4e-4
lossfunctions_considered = ["MRAE", "SAM", "SID"]			# Select which loss functions are considered for the reconstruction model training

### Variables used for printing the results in the logs ###
APPLICATION_NAME = "shelflife"
MODEL_NAME = "MST++"
CLASSIFIER_MODEL_NAME = "ModelWithAttention"
ILLUMINATIONS = ["H", "L", "C", "M"]

illumination_string = ""
if "H" in ILLUMINATIONS:
	illumination_string += "Halogen"
if "L" in ILLUMINATIONS:
	illumination_string += " + LED"
if "C" in ILLUMINATIONS:
	illumination_string += " + CFL"
if "M" in ILLUMINATIONS:
	illumination_string += " + Mixed"

model_run_title = "Model: %s\tDataset: %s\tIllumination: %s\tLosses: %s\tFull Image or Patches: %s\n" \
	% (MODEL_NAME, APPLICATION_NAME, illumination_string, lossfunctions_considered, "Full Image" if PATCH_SIZE == IMAGE_SIZE else "Patches")
classicication_run_title = "Model: %s\tDataset: %s\tNumber of Classes: %d\tSub Classes: %s\tFull Image or Patches: %s\tMobile Data\n" \
	% (CLASSIFIER_MODEL_NAME, APPLICATION_NAME, len(LABELS_DICT), len(TIME_LEFT_DICT), "Full Image" if PATCH_SIZE == IMAGE_SIZE else "Patches")

### to create the checkpoint of the model ###
checkpoint_fileprestring = "%s_%s" % (MODEL_NAME, APPLICATION_NAME)
classification_checkpoint_fileprestring = "%s_%s" % (CLASSIFIER_MODEL_NAME, APPLICATION_NAME)
checkpoint_file = "MSLP_%s_100.pkl" % checkpoint_fileprestring

# Mobile Model Files (Will delete these)
mobile_model_file = "model_%s.pth" % APPLICATION_NAME
onnx_file_name = "model.onnx"
tf_model_dir = os.path.join("tfmodel")
tflite_filename = "model.tflite"

### Formatting used for the visualizations ###
plt_dict = {"mathtext.default": "regular", "axes.linewidth": 2}
confusion_font_dict = {"family" : "serif", "weight": "normal", "size" : 45}
text_font_dict = {"family": "serif", "size": 30}
title_font_dict = {"fontname": "serif", "size": 40}

def sampler():
	global BANDS, NUMBER_OF_BANDS, BANDS_WAVELENGTHS
	sampled_bands = []

	for band_segment in range(len(BANDS)):
		sampled_bands.append(list(BANDS[band_segment]))

	BANDS = [band for sampled_list in sampled_bands for band in sampled_list]
	NUMBER_OF_BANDS = len(BANDS)
	BANDS_WAVELENGTHS = [int(round(BANDS_WAVELENGTHS[band])) for band in BANDS]
	return BANDS

sampler()

BANDS = [band for band in range(len(BANDS))] if use_mobile_dataset else BANDS

### Bands for text in the visualizations ###
VIEW_BANDS = [12, 23, 35, 46, 57, 67]
ACTUAL_BANDS = [round(BANDS_WAVELENGTHS[band], -2) for band in VIEW_BANDS]