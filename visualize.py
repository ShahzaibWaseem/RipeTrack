import os
import numpy as np
from glob import glob
from skimage import exposure

from utils import load_mat
from loss import test_psnr

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import ILLUMINATIONS, MODEL_NAME, TEST_ROOT_DATASET_DIR, TEST_DATASETS, VIEW_BANDS, ACTUAL_BANDS, var_name, text_font_dict, title_font_dict, plt_dict, init_directories

def main():
	for test_dataset in TEST_DATASETS:
		for illumination in ILLUMINATIONS:
			TEST_DATASET_DIR = os.path.join(TEST_ROOT_DATASET_DIR, "working_%s" % test_dataset, "%s_%s_204ch" % (test_dataset, illumination), "test")
				
			GT_PATH = os.path.join(TEST_DATASET_DIR, "mat")
			PLOTS_PATH = os.path.join(TEST_DATASET_DIR, "images")
			INF_PATH = os.path.join(TEST_DATASET_DIR, "inference")

			print("\nDataset: %s\nIllumination: %s\n" % (test_dataset, illumination))

			for filename in glob(os.path.join(INF_PATH, "resnext", "*.mat")):
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
					axs[0, j].set_title(str(ACTUAL_BANDS[j]) + " nm", **title_font_dict)
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
					axs[2, j].text(75, 570, "PSNR=%0.2f" % test_psnr(img, lab), **text_font_dict)
					axs[2, j].set_xticks([])
					axs[2, j].set_yticks([])

				norm = matplotlib.colors.Normalize(0, 1)
				# divider = make_axes_locatable(plt.gca())
				# cax = divider.append_axes("right", "5%", pad="1%")
				cax = fig.add_axes([0.945, 0.0455, 0.015, 0.2945])
				cax.tick_params(labelsize=18)
				matplotlib.colorbar.ColorbarBase(cax, cmap=plt.get_cmap("hot_r"), norm=norm)
				axs[0, len(VIEW_BANDS) - 1].text(525, 325, "$MS_{51}$", rotation=-90, **text_font_dict)
				axs[1, len(VIEW_BANDS) - 1].text(525, 425, "Ground Truth", rotation=-90, **text_font_dict)
				fig.tight_layout(pad=1, h_pad=-1.5, w_pad=-5)
				fig.savefig(os.path.join(PLOTS_PATH, MODEL_NAME, "%s.png" % (gt_filename)), dpi=fig.dpi*2, bbox_inches="tight")
				plt.show()

if __name__ == "__main__":
	plt.rcParams.update(plt_dict)
	init_directories()
	main()