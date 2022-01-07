import os
import numpy as np
from glob import glob
from skimage import exposure

from utils import load_mat
from loss import test_psnr

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import ILLUMINATIONS, TEST_ROOT_DATASET_DIR, TEST_DATASETS, VIEW_BANDS, ACTUAL_BANDS, var_name, fusion_techniques, init_directories

def main():
	for fusion in fusion_techniques:
		for test_dataset in TEST_DATASETS:
			for illumination in ILLUMINATIONS:
				TEST_DATASET_DIR = os.path.join(TEST_ROOT_DATASET_DIR, "working_%s" % test_dataset, "%s_%s_204ch" % (test_dataset, illumination), "test")
				
				GT_PATH = os.path.join(TEST_DATASET_DIR, "mat")
				PLOTS_PATH = os.path.join(TEST_DATASET_DIR, "images")
				INF_PATH = os.path.join(TEST_DATASET_DIR, "inference")

				print("\nFusion: %s\nDataset: %s\nIllumination: %s\n" % (fusion, test_dataset, illumination))

				for filename in glob(os.path.join(INF_PATH, fusion, "*.mat")):
					if(illumination == "cfl_led"):
						gt_filename = "_".join(filename.split("/")[-1].split(".")[0].split("_")[1:3])
					else:
						gt_filename = filename.split("/")[-1].split(".")[0].split("_")[-1]

					print(gt_filename + ".mat")
					
					inf_file = load_mat(filename, var_name)[var_name]
					gt_file = load_mat(os.path.join(GT_PATH, gt_filename + ".mat"), var_name)[var_name]
					gt_file = gt_file[:,:,1:204:4]

					fig, axs = plt.subplots(nrows=3, ncols=len(VIEW_BANDS), figsize=(15, 11))
					
					for j in range(axs.shape[1]):
						# Reconstructed Hypercube (gamma adjustment)
						img = inf_file[:,:,VIEW_BANDS[j]].reshape(512, 512)
						axs[0, j].imshow(exposure.adjust_gamma(img, 0.25), interpolation="nearest", cmap="gray")
						axs[0, j].set_title(str(ACTUAL_BANDS[j]) + " nm", size=16)
						axs[0, j].set_xticks([])
						axs[0, j].set_yticks([])

						# Ground Truth Hypercube (gamma adjustment)
						lab = gt_file[:,:,VIEW_BANDS[j]].reshape(512, 512)
						axs[1, j].imshow(exposure.adjust_gamma(lab, 0.25), interpolation="nearest", cmap="gray")
						axs[1, j].set_xticks([])
						axs[1, j].set_yticks([])

						# Difference b/w the two hypercubes
						diff = np.abs(lab - img)
						axs[2, j].imshow(diff, cmap="hot_r")
						axs[2, j].text(150, 550, "PSNR=%0.2f" % test_psnr(img, lab), size=16)
						axs[2, j].set_xticks([])
						axs[2, j].set_yticks([])

					norm = matplotlib.colors.Normalize(0, 1)
					# divider = make_axes_locatable(plt.gca())
					# cax = divider.append_axes("right", "5%", pad="1%")
					cax = fig.add_axes([0.963, 0.0449, 0.01, 0.285])
					matplotlib.colorbar.ColorbarBase(cax, cmap=plt.get_cmap("hot_r"), norm=norm)
					axs[0, len(VIEW_BANDS) - 1].text(525, 370, "Reconstructed", size=16, rotation=-90)
					axs[1, len(VIEW_BANDS) - 1].text(525, 370, "Ground Truth", size=16, rotation=-90)
					fig.tight_layout(pad=2)
					fig.savefig(os.path.join(PLOTS_PATH, fusion, "%s.png" % (gt_filename)), dpi=fig.dpi*2)
					plt.show()

if __name__ == "__main__":
	init_directories()
	main()