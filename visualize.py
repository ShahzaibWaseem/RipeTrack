import os
import numpy as np
from glob import glob
from utils import load_mat
import matplotlib.pyplot as plt

from config import ILLUMINATIONS, TEST_ROOT_DATASET_DIR, TEST_DATASETS, VIEW_BANDS, var_name, fusion_techniques, init_directories

def main():
	for fusion in fusion_techniques:
		for test_dataset in TEST_DATASETS:
			for illumination in ILLUMINATIONS:
				TEST_DATASET_DIR = os.path.join(TEST_ROOT_DATASET_DIR, "working_%s" % test_dataset, "%s_%s_204ch" % (test_dataset, illumination), "test")
				
				GT_PATH = os.path.join(TEST_DATASET_DIR, "mat")
				PLOTS_PATH = os.path.join(TEST_DATASET_DIR, "images")
				INF_PATH = os.path.join(TEST_DATASET_DIR, "inference")

				print("Fusion: %s\nDataset: %s\nIllumination: %s" % (fusion, test_dataset, illumination))

				for filename in glob(os.path.join(INF_PATH, fusion, "*.mat")):
					gt_filename = filename.split("/")[-1].split(".")[0].split("_")[-1]
					print(gt_filename + ".mat")

					inf_file = load_mat(filename, var_name)[var_name]
					gt_file = load_mat(os.path.join(GT_PATH, gt_filename+".mat"), var_name)[var_name]

					# view a section of the mat file created and compare it to the ground truth
					if "195" in filename:
						print(inf_file[1:10,1:10,0])
						print(gt_file[1:10,1:10,0])

					fig, axs = plt.subplots(nrows=len(VIEW_BANDS), ncols=2, figsize=(5, 15))
					for i, ax in enumerate(axs):
						img = inf_file[:,:,VIEW_BANDS[i]].reshape(512, 512)
						ax[0].imshow(img, interpolation="none")
						ax[0].set_title("Band:" + str(VIEW_BANDS[i]))

						lab = gt_file[:,:,VIEW_BANDS[i]].reshape(512, 512)
						ax[1].imshow(lab, interpolation="none")
						ax[1].set_title("Ground Truth")

					fig.savefig(os.path.join(PLOTS_PATH, fusion, "%s.png" % (gt_filename)), dpi=fig.dpi*2)
					plt.show()

if __name__ == "__main__":
	init_directories()
	main()