import os, cv2
import numpy as np
import hdf5storage as h5io

from heatmap_overlay import *



def visual_color_img(RootDir, MethodNames, with_fix=0):

	imgsDir = RootDir + 'images/'
	fixsDir = RootDir + 'fixations/maps/'
	salsDir = RootDir + 'Results/Saliency/'

	img_ext = '.jpg'
	sal_ext = '.png'

	for idx_m in range(len(MethodNames)):
		print("---" + str(idx_m + 1) + "/" + str(len(MethodNames)) + "---: " + MethodNames[idx_m])

		salmap_dir = salsDir + MethodNames[idx_m] + '/'
		out_path = salmap_dir + 'Visual_color/'
		if not os.path.exists(out_path):
			os.makedirs(out_path)

		sal_names = [f for f in os.listdir(salmap_dir) if f.endswith(sal_ext)]
		sal_names.sort()

		for idx_n in range(len(sal_names)):
			print(str(idx_n + 1) + "/" + str(len(sal_names)) + ": " + sal_names[idx_n])

			file_name = sal_names[idx_n][:-4]
			outname = out_path + file_name + sal_ext
			if os.path.exists(outname):
				continue

			img    = cv2.imread(imgsDir + file_name + img_ext,-1)
			salmap = cv2.imread(salmap_dir + file_name + sal_ext,-1)

			fixname = fixsDir + file_name + '.mat'
			if with_fix and os.path.exists(fixname):
				fixmap = h5io.loadmat(fixname)["I"]

			overmap = heatmap_overlay(img,salmap)
			if with_fix and os.path.exists(fixname):
				fixpts_dilate = cv2.dilate(fixmap, np.ones((5, 5), np.uint8))
				fixpts_dilate = np.repeat(np.expand_dims(fixpts_dilate, axis=2), 3, axis=2)
				overmap[fixpts_dilate > 0.5] = 1

			overmap = overmap / np.max(overmap) *255
			cv2.imwrite(out_path + file_name + sal_ext, im2uint8(overmap))


if __name__ == "__main__":

	DataSet = 'salicon'
	RootDir = 'E:/Code/IIP_Saliency_Video/DataSet/salicon/val/'
	MethodNames = ['zk-twos-st']

	WITH_FIX = 0
	visual_color_img(RootDir, MethodNames, with_fix=WITH_FIX)

