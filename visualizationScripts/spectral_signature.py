import os, sys
sys.path.append(os.path.join(".."))

from imageio import imread
import matplotlib.pyplot as plt

from skimage import exposure

import matplotlib
from matplotlib.patches import Circle

from utils import load_mat
from config import BAND_SPACING, NORMALIZATION_FACTOR, TEST_ROOT_DATASET_DIR, text_font_dict

def main():
	rgb_img = imread(os.path.join(os.path.dirname(TEST_ROOT_DATASET_DIR), "1458.png"))
	# rgb_img = np.array(rgb_img).T

	gt_img = load_mat(os.path.join(TEST_ROOT_DATASET_DIR, "working_avocado", "avocado_h_204ch", "test", "mat", "1458.mat"))
	gt_img = gt_img[:, :, ::BAND_SPACING] / NORMALIZATION_FACTOR

	inf_img = load_mat(os.path.join(TEST_ROOT_DATASET_DIR, "working_avocado", "avocado_h_204ch", "test", "inference", "resnext", "inf_1458.mat")) / NORMALIZATION_FACTOR

	# hscnn_inf = load_mat(os.path.join("..", "HSCNN-R", "test", "inference", "HSCNN_4to51_steak_h", "inf_1885.mat")) / NORMALIZATION_FACTOR
	# awan_inf = load_mat(os.path.join("..", "AWAN", "test", "inference", "AWAN_4to51_steak_h", "inf_1885.mat")) / NORMALIZATION_FACTOR

	fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
	x=range(400, 1001, 12)
	xlim=[400, 1000]
	point1 = [350, 256]
	point2 = [64, 350]

	axs[0].plot(x, gt_img[point1[0], point1[1], :], "r--", label="Ground Truth (unripe)", linewidth=3)
	axs[0].plot(x, inf_img[point1[0], point1[1], :], "r-", label="MobiSLP (unripe)", linewidth=3)

	axs[0].plot(x, gt_img[point2[0], point2[1], :], "b--", label="Ground Truth (ripe)", linewidth=3)
	axs[0].plot(x, inf_img[point2[0], point2[1], :], "b-", label="MobiSLP (ripe)", linewidth=3)
	axs[0].set_xlim(xlim)
	axs[0].set_xlabel("Wavelength (nm)")
	axs[0].set_ylabel("Normalized Intensity")

	axs[0].legend()

	axs[1].imshow(exposure.adjust_gamma(rgb_img, 0.5))
	axs[1].set_xticks([])
	axs[1].set_yticks([])

	circle1 = Circle((point1[1], point1[0]), 15, edgecolor="r", fill=0, linestyle="--", linewidth=7)
	axs[1].add_patch(circle1)
	circle2 = Circle((point2[1], point2[0]), 15, edgecolor="b", fill=0, linestyle="--", linewidth=7)
	axs[1].add_patch(circle2)
	
	fig.tight_layout(w_pad=-2.5)
	plt.savefig(os.path.join("logs", "signature_avocado.pdf"), dpi=fig.dpi*2, bbox_inches="tight")

if __name__ == "__main__":
	os.chdir("..")
	matplotlib.rc("font", **text_font_dict)
	main()