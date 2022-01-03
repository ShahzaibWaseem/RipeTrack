import os

TRAIN_DATASET_DIR = os.path.join("..", "data_preparation")
TRAIN_DATASET_FILES = ["train_chicken_halogen_4to51bands.h5",
					   "train_steak_halogen_4to51bands.h5",
					   "train_steak_cfl_led_4to51bands.h5"]
VALID_DATASET_FILES = ["valid_chicken_halogen_4to51bands.h5",
					   "valid_steak_halogen_4to51bands.h5",
					   "valid_steak_cfl_led_4to51bands.h5"]

batch_size = 64
end_epoch = 50
init_lr = 0.0001
fusion = "concat"

MODEL_PATH = os.path.join(".", "checkpoints")
LOGS_PATH = os.path.join(".", "logs")


TEST_DATASET_DIR = os.path.join("..", "working_apples", "apple_h_204ch", "test")
GT_PATH = os.path.join(TEST_DATASET_DIR, "mat")
IMG_PATH = os.path.join(TEST_DATASET_DIR, "cameraRGBN")
INF_PATH = os.path.join(".", "inference")

checkpoint_file = "HS_model_49.pkl"
# checkpoint_file = "HS_model_%d.pkl" % end_epoch

VIEW_BANDS = [15, 17, 19, 21, 24]

var_name = "rad"

def init_directories():
	for directory in [MODEL_PATH, LOGS_PATH, INF_PATH]:
		if not os.path.exists(directory):
			os.makedirs(directory)