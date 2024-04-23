import os
import sys
sys.path.append(os.path.join(".."))

from imageio import imread
from skimage import exposure

from utils import load_mat
from config import BANDS, TEST_ROOT_DATASET_DIR, VISUALIZATION_DIR_NAME, GT_HYPERCUBES_DIR_NAME, RECONSTRUCTED_HS_DIR_NAME, text_font_dict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

matplotlib.rc("font", **text_font_dict)

def main():
	rgb_img = imread(os.path.join(os.path.dirname(TEST_ROOT_DATASET_DIR), "1458.png"))
	# rgb_img = np.array(rgb_img).T

	gt_hypercube = load_mat(os.path.join(TEST_ROOT_DATASET_DIR, "%s_204ch" % "avocado-hass", GT_HYPERCUBES_DIR_NAME, "1458.mat"))
	gt_hypercube = gt_hypercube[:,:,BANDS]
	gt_hypercube = (gt_hypercube - gt_hypercube.min()) / (gt_hypercube.max() - gt_hypercube.min())

	inf_hypercube = load_mat(os.path.join(TEST_ROOT_DATASET_DIR, "%s_204ch" % "avocado-hass", RECONSTRUCTED_HS_DIR_NAME, "1458.mat"))

	fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
	x=range(400, 1001, 12)
	xlim=[400, 1000]
	point1 = [350, 256]
	point2 = [64, 350]

	axs[0].plot(x, gt_hypercube[point1[0], point1[1], :], "r--", label="Ground Truth (unripe)", linewidth=3)
	axs[0].plot(x, inf_hypercube[point1[0], point1[1], :], "r-", label="MobiSLP (unripe)", linewidth=3)

	axs[0].plot(x, gt_hypercube[point2[0], point2[1], :], "b--", label="Ground Truth (ripe)", linewidth=3)
	axs[0].plot(x, inf_hypercube[point2[0], point2[1], :], "b-", label="MobiSLP (ripe)", linewidth=3)
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
	plt.savefig(os.path.join(VISUALIZATION_DIR_NAME, "signature_avocado.pdf"), dpi=fig.dpi*2, bbox_inches="tight")

if __name__ == "__main__":
	os.chdir("..")
	main()