import os
import numpy as np
from glob import glob
from utils import load_mat
import matplotlib.pyplot as plt
from config import GT_PATH, INF_PATH, VIEW_BANDS, var_name

def main():
	for filename in glob(os.path.join(INF_PATH, "*.mat")):
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

		fig.savefig(os.path.join("images", "%s.png" % (gt_filename)), dpi=fig.dpi*2)
		plt.show()

if __name__ == "__main__":
	main()