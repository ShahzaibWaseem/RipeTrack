import os, sys
sys.path.append(os.path.join(".."))

from glob import glob
from skimage import exposure

from utils import load_mat
from loss import test_psnr

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from config import BAND_SPACING, ILLUMINATIONS, MODEL_NAME, TEST_ROOT_DATASET_DIR, TEST_DATASETS, VIEW_BANDS, ACTUAL_BANDS, text_font_dict, title_font_dict, plt_dict, init_directories

def main():
	for test_dataset in TEST_DATASETS:
		for illumination in ILLUMINATIONS:
			TEST_DATASET_DIR = os.path.join(TEST_ROOT_DATASET_DIR, "working_%s" % test_dataset, "%s_%s_204ch" % (test_dataset, illumination), "test")

			GT_PATH = os.path.join(TEST_DATASET_DIR, "mat")
			PLOTS_PATH = os.path.join(TEST_DATASET_DIR, "images")
			INF_PATH = os.path.join(TEST_DATASET_DIR, "inference")
			cfl_file = load_mat(os.path.join(TEST_ROOT_DATASET_DIR, "working_%s" % test_dataset, "%s_nh_204ch" % test_dataset, "1924.mat"))
			cfl_file = cfl_file[:,:, ::BAND_SPACING]

			print("\nDataset: %s\nIllumination: %s\n" % (test_dataset, illumination))

			for filename in sorted(glob(os.path.join(INF_PATH, "resnext", "*1925_f.mat"))):
				if(illumination == "cfl_led"):
					gt_filename = "_".join(filename.split("/")[-1].split(".")[0].split("_")[1:3])
				else:
					gt_filename = filename.split("/")[-1].split(".")[0].split("_")[-1]

				print(gt_filename + ".mat")

				inf_file = load_mat(filename)
				gt_file = load_mat(os.path.join(GT_PATH, gt_filename + ".mat"))
				gt_file = gt_file[:,:, ::BAND_SPACING]

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
					# diff = np.abs(lab - img)
					# psnr = test_psnr(lab, img)	"PSNR=%.2f" % psnr	cmap="hot_r"

					cfl = cfl_file[:,:,VIEW_BANDS[j]].reshape(512, 512)
					axs[2, j].imshow(exposure.adjust_gamma(cfl, 0.25), interpolation="nearest", cmap="gray")
					axs[2, j].set_xlabel(" ", loc="center", fontsize=25)
					axs[2, j].set_xticks([])
					axs[2, j].set_yticks([])

				# norm = matplotlib.colors.Normalize(0, 1)
				# axin = inset_axes(axs[2, len(VIEW_BANDS) - 1], height="100%", width="5%", loc="right", borderpad=-1.5)
				# axin.tick_params(labelsize=18)
				# matplotlib.colorbar.ColorbarBase(axin, cmap=plt.get_cmap("hot_r"), norm=norm)
				axs[0, len(VIEW_BANDS) - 1].yaxis.set_label_position("right")
				axs[0, len(VIEW_BANDS) - 1].set_ylabel("MobiSpectral", loc="center", rotation=-90, labelpad=30, **text_font_dict)
				axs[1, len(VIEW_BANDS) - 1].yaxis.set_label_position("right")
				axs[1, len(VIEW_BANDS) - 1].set_ylabel("Ground Truth", loc="center", rotation=-90, labelpad=30, **text_font_dict)
				axs[2, len(VIEW_BANDS) - 1].yaxis.set_label_position("right")
				axs[2, len(VIEW_BANDS) - 1].set_ylabel("CFL", loc="center", rotation=-90, labelpad=30, **text_font_dict)

				fig.tight_layout(pad=1, h_pad=1, w_pad=-5)
				fig.savefig(os.path.join(PLOTS_PATH, MODEL_NAME, "%s.pdf" % (gt_filename)), dpi=fig.dpi*2, bbox_inches="tight")
				plt.show()

if __name__ == "__main__":
	os.chdir("..")
	plt.rcParams.update(plt_dict)
	init_directories()
	main()