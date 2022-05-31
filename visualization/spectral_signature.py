import os, sys
sys.path.append(os.path.join(".."))

from imageio import imread
import matplotlib.pyplot as plt

from skimage import exposure

import matplotlib
from matplotlib.patches import Circle

from utils import load_mat
from config import TEST_ROOT_DATASET_DIR, text_font_dict

def main():
	rgb_img = imread(os.path.join(os.path.dirname(TEST_ROOT_DATASET_DIR), "1458.png"))
	# rgb_img = np.array(rgb_img).T

	gt_img = load_mat(os.path.join(TEST_ROOT_DATASET_DIR, "working_avocado", "avocado_h_204ch", "test", "mat", "1458.mat"))
	gt_img = gt_img[:, :, 1:204:4] / 4096

	inf_img = load_mat(os.path.join(TEST_ROOT_DATASET_DIR, "working_avocado", "avocado_h_204ch", "test", "inference", "resnext", "inf_1458.mat")) / 4096

	# hscnn_inf = load_mat(os.path.join("..", "HSCNN-R", "test", "inference", "HSCNN_4to51_steak_h", "inf_1885.mat")) / 4096
	# awan_inf = load_mat(os.path.join("..", "AWAN", "test", "inference", "AWAN_4to51_steak_h", "inf_1885.mat")) / 4096

	fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
	x=range(400, 1001, 12)
	xlim=[400, 1000]

	axs[0].plot(x, gt_img[350, 256, :], "r--", label="Ground Truth (unripe)", linewidth=3)
	axs[0].plot(x, inf_img[350, 256, :], "r-", label="MobiSpectral (unripe)", linewidth=3)

	axs[0].plot(x, gt_img[64, 350, :], "b--", label="Ground Truth (ripe)", linewidth=3)
	axs[0].plot(x, inf_img[64, 350, :], "b-", label="MobiSpectral (ripe)", linewidth=3)
	axs[0].set_xlim(xlim)
	axs[0].set_xlabel("Wavelength (nm)")
	axs[0].set_ylabel("Normalized Intensity")

	axs[0].legend()

	axs[1].imshow(exposure.adjust_gamma(rgb_img, 0.5))
	axs[1].set_xticks([])
	axs[1].set_yticks([])

	circle1 = Circle((256, 350), 15, edgecolor="r", fill=0, linestyle="--", linewidth=7)
	axs[1].add_patch(circle1)
	circle2 = Circle((64, 350), 15, edgecolor="b", fill=0, linestyle="--", linewidth=7)
	axs[1].add_patch(circle2)
	
	fig.tight_layout(w_pad=-2.5)
	plt.savefig(os.path.join("logs", "signature_avocado.pdf"), dpi=fig.dpi*2, bbox_inches="tight")

if __name__ == "__main__":
	os.chdir("..")
	matplotlib.rc("font", **text_font_dict)
	main()