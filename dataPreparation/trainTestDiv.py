import os
import sys
sys.path.append("..")
from glob import glob

import random

from config import TEST_ROOT_DATASET_DIR, APPLICATION_NAME, TEST_DATASETS, TRAIN_VAL_TEST_SPLIT_DIR_NAME, create_directory

def saveFile(filepath, data):
	with open(filepath, "w") as file:
		for item in data:
			file.write("%s\n" % item)

def main():
	train_size = 0.7
	validation_size = 0.15
	for dataset in TEST_DATASETS:
		dataset_items = []
		print("\nDataset:", dataset)

		for filename in glob(os.path.join("..", TEST_ROOT_DATASET_DIR, APPLICATION_NAME, "%s_204ch" % dataset, "*.mat")):
			filename_cleaned = int(os.path.split(filename)[-1].split(".")[0])
			dataset_items.append(filename_cleaned)
			random.shuffle(dataset_items)

		train_dataset = dataset_items[:int(len(dataset_items)*train_size)]
		validation_dataset = dataset_items[int(len(dataset_items)*train_size):int(len(dataset_items)*(train_size+validation_size))]
		test_dataset = dataset_items[int(len(dataset_items)*(train_size+validation_size)):]

		print(dataset_items)
		print("Total:", len(dataset_items))
		print("Train:", len(train_dataset))
		print("Validation:", len(validation_dataset))
		print("Test:", len(test_dataset))
		create_directory(os.path.join("..", TEST_ROOT_DATASET_DIR, APPLICATION_NAME, "%s_204ch" % dataset, TRAIN_VAL_TEST_SPLIT_DIR_NAME))

		saveFile(os.path.join("..", TEST_ROOT_DATASET_DIR, APPLICATION_NAME, "%s_204ch" % dataset, TRAIN_VAL_TEST_SPLIT_DIR_NAME, "train.txt"), sorted(train_dataset))
		saveFile(os.path.join("..", TEST_ROOT_DATASET_DIR, APPLICATION_NAME, "%s_204ch" % dataset, TRAIN_VAL_TEST_SPLIT_DIR_NAME, "validation.txt"), sorted(validation_dataset))
		saveFile(os.path.join("..", TEST_ROOT_DATASET_DIR, APPLICATION_NAME, "%s_204ch" % dataset, TRAIN_VAL_TEST_SPLIT_DIR_NAME, "test.txt"), sorted(test_dataset))

if __name__ == "__main__":
	main()