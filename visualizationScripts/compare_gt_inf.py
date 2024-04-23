import os
import sys
sys.path.append(os.path.join(".."))

import numpy as np
from glob import glob
from skimage import exposure

from loss import test_psnr
from utils import load_mat, create_directory
from config import GT_HYPERCUBES_DIR_NAME, RECONSTRUCTED_HS_DIR_NAME, VISUALIZATION_DIR_NAME,\
	TEST_DATASETS, TEST_ROOT_DATASET_DIR, BANDS, VIEW_BANDS, ACTUAL_BANDS, APPLICATION_NAME, IMAGE_SIZE,\
	text_font_dict, title_font_dict, plt_dict

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def main():
	for dataset in TEST_DATASETS:
		directory = os.path.join(TEST_ROOT_DATASET_DIR, APPLICATION_NAME, "{}_204ch".format(dataset), GT_HYPERCUBES_DIR_NAME)
		inf_directory = os.path.join(directory, RECONSTRUCTED_HS_DIR_NAME)
		print(" " * 19, "{0:62}".format(directory), RECONSTRUCTED_HS_DIR_NAME)
		create_directory(os.path.join(directory, VISUALIZATION_DIR_NAME))

		for filename in glob(os.path.join(directory, "664.mat")):
			inf_hypercube = load_mat(os.path.join(inf_directory, os.path.split(filename)[-1]))

			gt_hypercube = load_mat(filename)
			gt_hypercube = gt_hypercube[:,:,BANDS]
			gt_hypercube = (gt_hypercube - gt_hypercube.min()) / (gt_hypercube.max() - gt_hypercube.min())

			fig, axs = plt.subplots(nrows=3, ncols=len(VIEW_BANDS), figsize=(15, 11))

			for j in range(axs.shape[1]):
				# Reconstructed Hypercube (gamma adjustment)
				inf_band = inf_hypercube[:,:,VIEW_BANDS[j]].reshape(IMAGE_SIZE, IMAGE_SIZE)
				gt_band = gt_hypercube[:,:,VIEW_BANDS[j]].reshape(IMAGE_SIZE, IMAGE_SIZE)

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
				axs[2, j].set_xlabel("PSNR=%.2f" % psnr, **text_font_dict)

			# inserting colorbar showing the range of errors
			norm = matplotlib.colors.Normalize(0, 1)
			axin = inset_axes(axs[2, len(VIEW_BANDS) - 1], height="100%", width="7%", loc="right", borderpad=-1.5)
			axin.tick_params(labelsize=18)
			matplotlib.colorbar.ColorbarBase(axin, cmap=plt.get_cmap("hot_r"), norm=norm)

			# showing texts for axs
			axs[0, len(VIEW_BANDS) - 1].yaxis.set_label_position("right")
			axs[0, len(VIEW_BANDS) - 1].set_ylabel("MobiSLP", loc="center", rotation=-90, labelpad=30, **text_font_dict)
			axs[1, len(VIEW_BANDS) - 1].yaxis.set_label_position("right")
			axs[1, len(VIEW_BANDS) - 1].set_ylabel("Ground Truth", loc="center", rotation=-90, labelpad=30, **text_font_dict)
			axs[2, len(VIEW_BANDS) - 1].yaxis.set_label_position("right")
			# axs[2, len(VIEW_BANDS) - 1].set_ylabel("Error Map", loc="center", rotation=-90, labelpad=30, **text_font_dict)

			fig.tight_layout(pad=1, h_pad=1, w_pad=-5)
			fig.savefig(os.path.join(directory, VISUALIZATION_DIR_NAME, "%s.pdf" % (os.path.split(filename)[-1].split(".")[0])), dpi=fig.dpi*2, bbox_inches="tight")
			plt.show()

if __name__ == "__main__":
	os.chdir("..")
	plt.rcParams.update(plt_dict)
	main()