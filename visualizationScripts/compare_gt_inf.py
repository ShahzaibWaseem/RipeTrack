import os
import sys
sys.path.append(os.path.join(".."))

import numpy as np
from glob import glob
from imageio import imread
from skimage import exposure

from loss import test_psnr
from utils import load_mat, create_directory
from config import GT_HYPERCUBES_DIR_NAME, RECONSTRUCTED_HS_DIR_NAME, GT_RGBN_DIR_NAME, VISUALIZATION_DIR_NAME,\
	TRAIN_VAL_TEST_SPLIT_DIR_NAME, TEST_ROOT_DATASET_DIR, BANDS, VIEW_BANDS, ACTUAL_BANDS, APPLICATION_NAME,\
	EPS, text_font_dict, title_font_dict, plt_dict

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

hypercubes_considered = ["483.mat", "887.mat"]
crops = {"887.mat": [50, 350, 100, 400], "483.mat": [50, 350, 150, 450]}

def main():
	for dataset in ["pear-bartlett", "avocado-organic"]:
		directory = os.path.join(TEST_ROOT_DATASET_DIR, APPLICATION_NAME, "{}_204ch".format(dataset))
		inf_directory = os.path.join(directory, RECONSTRUCTED_HS_DIR_NAME)
		print(" " * 19, "{0:62}".format(directory), RECONSTRUCTED_HS_DIR_NAME)
		create_directory(os.path.join(directory, VISUALIZATION_DIR_NAME))
		hypercube_filename = list(crops.keys())[1]

		# with open(os.path.join(directory, TRAIN_VAL_TEST_SPLIT_DIR_NAME, "test.txt"), "r") as test_file:
		# 	hypercube_list = [filename.replace("\n", ".mat") for filename in test_file]

		# for filename in hypercube_list:
		for filename in glob(os.path.join(directory, GT_HYPERCUBES_DIR_NAME, hypercube_filename)):
			hypercube_number = os.path.split(filename)[-1].split(".")[0]
			print("Processing", hypercube_number)
			xmin, xmax, ymin, ymax = crops[hypercube_number + ".mat"]

			inf_hypercube = load_mat(os.path.join(inf_directory, os.path.split(filename)[-1]))
			inf_hypercube = inf_hypercube[xmin:xmax, ymin:ymax, :]

			gt_hypercube = load_mat(filename)
			gt_hypercube = gt_hypercube[:, :, BANDS]
			gt_hypercube = (gt_hypercube - gt_hypercube.min()) / (gt_hypercube.max() - gt_hypercube.min())
			gt_hypercube = gt_hypercube[xmin:xmax, ymin:ymax, :]
			gt_hypercube = gt_hypercube + EPS

			rgb_image = imread(os.path.join(directory, GT_RGBN_DIR_NAME, hypercube_number + "_RGB.png"))
			rgb_image = rgb_image[xmin:xmax, ymin:ymax, :]

			fig, axs = plt.subplots(nrows=3, ncols=len(VIEW_BANDS), figsize=(15, 8))

			for j in range(axs.shape[1]):
				# Reconstructed Hypercube (gamma adjustment)
				inf_band = inf_hypercube[:, :, VIEW_BANDS[j]]
				gt_band = gt_hypercube[:, :, VIEW_BANDS[j]]

				# Difference b/w the two hypercubes
				diff_band = np.abs(gt_band - inf_band)
				psnr = test_psnr(gt_band, inf_band)

				inf_band = exposure.adjust_gamma(inf_band, 0.25)
				axs[0, j].imshow(inf_band, interpolation="nearest", cmap="gray")
				axs[0, j].set_title(str(ACTUAL_BANDS[j]) + " nm", **title_font_dict)
				axs[0, j].set_xticks([])
				axs[0, j].set_yticks([])

				# Ground Truth Hypercube (gamma adjustment)
				gt_band = exposure.adjust_gamma(gt_band, 0.25)
				axs[1, j].imshow(gt_band, interpolation="nearest", cmap="gray")
				axs[1, j].set_xticks([])
				axs[1, j].set_yticks([])

				axs[2, j].imshow(diff_band, interpolation="nearest", cmap="hot_r")
				axs[2, j].set_xticks([])
				axs[2, j].set_yticks([])
				# axs[2, j].set_xlabel("PSNR=%.2f" % psnr, **text_font_dict)

			# inserting colorbar showing the range of errors
			norm = matplotlib.colors.Normalize(0, 1)
			axin = inset_axes(axs[2, len(VIEW_BANDS) - 1], height="100%", width="7%", loc="right", borderpad=-1.5)
			axin.tick_params(labelsize=18)
			matplotlib.colorbar.ColorbarBase(axin, cmap=plt.get_cmap("hot_r"), norm=norm)

			# showing texts for axs
			axs[0, len(VIEW_BANDS) - 1].yaxis.set_label_position("right")
			axs[0, len(VIEW_BANDS) - 1].set_ylabel("RipeTrack", loc="center", rotation=-90, labelpad=30, **text_font_dict)
			axs[1, len(VIEW_BANDS) - 1].yaxis.set_label_position("right")
			axs[1, len(VIEW_BANDS) - 1].set_ylabel("GT", loc="center", rotation=-90, labelpad=30, **text_font_dict)
			axs[2, len(VIEW_BANDS) - 1].yaxis.set_label_position("right")
			# axs[2, len(VIEW_BANDS) - 1].set_ylabel("Error Map", loc="center", rotation=-90, labelpad=30, **text_font_dict)
			fig.tight_layout(pad=1)
			fig.savefig(os.path.join(directory, VISUALIZATION_DIR_NAME, "%s.pdf" % hypercube_number), dpi=fig.dpi*2, bbox_inches="tight")
			plt.show()

if __name__ == "__main__":
	os.chdir("..")
	plt.rcParams.update(plt_dict)
	main()