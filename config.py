import os

TRAIN_DATASET_DIR = os.path.join("..", "data_preparation", "datasets")
TRAIN_DATASET_FILES = ["train_chicken_halogen_4to51bands.h5",
					   "train_steak_halogen_4to51bands.h5",
					   "train_steak_cfl_led_4to51bands.h5"]
VALID_DATASET_FILES = ["valid_chicken_halogen_4to51bands.h5",
					   "valid_steak_halogen_4to51bands.h5",
					   "valid_steak_cfl_led_4to51bands.h5"]

batch_size = 64
end_epoch = 50
init_lr = 0.0001
fusion_techniques = ["concat", "add", "multiply"]

MODEL_PATH = os.path.join(".", "checkpoints")
LOGS_PATH = os.path.join(".", "logs")

TEST_ROOT_DATASET_DIR = os.path.join(os.path.dirname(TRAIN_DATASET_DIR), "working_datasets")
TEST_DATASETS = ["chicken", "steak"]
ILLUMINATIONS = ["h", "cfl_led"]

checkpoint_file = "HS_model_49.pkl"
# checkpoint_file = "HS_model_%d.pkl" % end_epoch

VIEW_BANDS = [10, 20, 30, 40, 50]

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
				for fusion in fusion_techniques:
					test_dataset_path = os.path.join(TEST_ROOT_DATASET_DIR, "working_%s" % test_dataset, "%s_%s_204ch" % (test_dataset, illumination), "test", directory, fusion)
					if not os.path.exists(test_dataset_path):
						os.makedirs(test_dataset_path)