import os, cv2
import numpy as np
import hdf5storage as h5io

from heatmap_overlay import *



def visual_color_vid(RootDir, DataSet, MethodNames, with_fix=0):

	vidsDir = RootDir + 'Videos/'
	fixsDir = RootDir + 'fixations/maps/'
	salsDir = RootDir + 'Results/Saliency/'

	vid_ext = '.mp4'
	if DataSet.upper() == 'CITIUS':
		vid_ext = '.avi'


	for idx_m in range(len(MethodNames)):
		print("---" + str(idx_m + 1) + "/" + str(len(MethodNames)) + "---: " + MethodNames[idx_m])

		salmap_dir = salsDir + MethodNames[idx_m] + '/'

		out_path = salmap_dir + 'Visual_color/'


		if not os.path.exists(out_path):
			os.makedirs(out_path)

		sal_names = [f for f in os.listdir(salmap_dir) if f.endswith('.mat')]
		sal_names.sort()

		for idx_n in range(len(sal_names)):
			print(str(idx_n + 1) + "/" + str(len(sal_names)) + ": " + sal_names[idx_n])

			file_name = sal_names[idx_n][:-4]
			outname = out_path + file_name + '.mp4'
			if os.path.exists(outname):
				continue

			VideoCap = cv2.VideoCapture(vidsDir + file_name + vid_ext)
			vidsize = (int(VideoCap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(VideoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
			vidframes = int(VideoCap.get(cv2.CAP_PROP_FRAME_COUNT))
			vidfps = VideoCap.get(cv2.CAP_PROP_FPS)

			salmap = np.rint(h5io.loadmat(salmap_dir + file_name + '.mat')["salmap"]).astype(np.uint8)

			nframes = min(vidframes, salmap.shape[3])
			fixname = fixsDir + file_name + '_fixPts.mat'
			if with_fix and os.path.exists(fixname):
				fixpts = h5io.loadmat(fixname)["fixLoc"]
				nframes = min(nframes, fixpts.shape[3])

			fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
			VideoWriter = cv2.VideoWriter(outname, fourcc, vidfps, vidsize, isColor=True)

			for idx_f in range(nframes):

				ret, img = VideoCap.read()
				isalmap = salmap[:, :, 0, idx_f]

				iovermap = heatmap_overlay(img,isalmap)

				if with_fix and os.path.exists(fixname):
					ifixpts = fixpts[:, :, 0, idx_f]
					ifixpts_dilate = cv2.dilate(ifixpts,np.ones((5,5), np.uint8))
					ifixpts_dilate = np.repeat(np.expand_dims(ifixpts_dilate, axis=2), 3, axis=2)
					iovermap[ifixpts_dilate>0.5] = 1

				iovermap = iovermap / np.max(iovermap) *255
				VideoWriter.write(im2uint8(iovermap))

			VideoCap.release()
			VideoWriter.release()

def visual_gray_vid(RootDir, DataSet, MethodNames, with_fix=0):

	fixsDir = RootDir + 'fixations/maps/'
	salsDir = RootDir + 'Results/Saliency/'

	vid_ext = '.mp4'
	if DataSet.upper() == 'CITIUS':
		vid_ext = '.avi'

	for idx_m in range(len(MethodNames)):
		print("---" + str(idx_m + 1) + "/" + str(len(MethodNames)) + "---: " + MethodNames[idx_m])

		salmap_dir = salsDir + MethodNames[idx_m] + '/'
		out_path = salmap_dir + 'Visual_gray/'

		if not os.path.exists(out_path):
			os.makedirs(out_path)

		sal_names = [f for f in os.listdir(salmap_dir) if f.endswith('.mat')]
		sal_names.sort()

		for idx_n in range(len(sal_names)):
			print(str(idx_n + 1) + "/" + str(len(sal_names)) + ": " + sal_names[idx_n])

			file_name = sal_names[idx_n][:-4]
			outname = out_path + file_name + '.mp4'
			if os.path.exists(outname):
				continue

			salmap = np.rint(h5io.loadmat(salmap_dir + file_name + '.mat')["salmap"]).astype(np.uint8)

			nframes = salmap.shape[3]
			fixname = fixsDir + file_name + '_fixPts.mat'
			if with_fix and os.path.exists(fixname):
				fixpts = h5io.loadmat(fixname)["fixLoc"]
				nframes = min(nframes, fixpts.shape[3])

			fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
			VideoWriter = cv2.VideoWriter(outname, fourcc, 25, (salmap.shape[1],salmap.shape[0]), isColor=True)

			for idx_f in range(nframes):

				isalmap = salmap[:, :, 0, idx_f]
				# iovermap = isalmap/255
				iovermap = np.repeat(np.expand_dims(isalmap, axis=2), 3, axis=2)/255

				if with_fix and os.path.exists(fixname):
					ifixpts = fixpts[:, :, 0, idx_f]
					ifixpts_dilate = cv2.dilate(ifixpts, np.ones((5, 5), np.uint8))
					ifixpts_dilate = np.repeat(np.expand_dims(ifixpts_dilate, axis=2), 3, axis=2)
					iovermap[ifixpts_dilate > 0.5] = 1

				iovermap = iovermap / np.max(iovermap) * 255
				VideoWriter.write(im2uint8(iovermap))

			VideoWriter.release()



def visual_vid(RootDir, DataSet, MethodNames, with_color=0, with_fix=0):

	vidsDir = RootDir + 'Videos/'
	fixsDir = RootDir + 'fixations/maps/'
	salsDir = RootDir + 'Results/Saliency/'

	vid_ext = '.mp4'
	if DataSet.upper() == 'CITIUS':
		vid_ext = '.avi'


	for idx_m in range(len(MethodNames)):
		print("---" + str(idx_m + 1) + "/" + str(len(MethodNames)) + "---: " + MethodNames[idx_m])

		salmap_dir = salsDir + MethodNames[idx_m] + '/'
		if with_color:
			out_path = salmap_dir + 'Visual_color/'
		else:
			out_path = salmap_dir + 'Visual_gray/'

		if not os.path.exists(out_path):
			os.makedirs(out_path)

		sal_names = [f for f in os.listdir(salmap_dir) if f.endswith('.mat')]
		sal_names.sort()

		for idx_n in range(len(sal_names)):
			print(str(idx_n + 1) + "/" + str(len(sal_names)) + ": " + sal_names[idx_n])

			file_name = sal_names[idx_n][:-4]
			outname = out_path + file_name + '.mp4'
			if os.path.exists(outname):
				continue

			VideoCap = cv2.VideoCapture(vidsDir + file_name + vid_ext)
			vidsize = (int(VideoCap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(VideoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
			vidframes = int(VideoCap.get(cv2.CAP_PROP_FRAME_COUNT))
			vidfps = VideoCap.get(cv2.CAP_PROP_FPS)

			salmap = np.rint(h5io.loadmat(salmap_dir + file_name + '.mat')["salmap"]).astype(np.uint8)

			nframes = min(vidframes, salmap.shape[3])
			fixname = fixsDir + file_name + '_fixPts.mat'
			if with_fix and os.path.exists(fixname):
				fixpts = h5io.loadmat(fixname)["fixLoc"]
				nframes = min(nframes, fixpts.shape[3])

			fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
			VideoWriter = cv2.VideoWriter(outname, fourcc, vidfps, vidsize, isColor=True)

			for idx_f in range(nframes):

				isalmap = salmap[:, :, 0, idx_f]

				if with_color:
					ret, img = VideoCap.read()
					iovermap = heatmap_overlay(img, isalmap)
				else:
					iovermap = np.repeat(np.expand_dims(isalmap, axis=2), 3, axis=2)/255

				if with_fix and os.path.exists(fixname):
					ifixpts = fixpts[:, :, 0, idx_f]
					ifixpts_dilate = cv2.dilate(ifixpts,np.ones((5,5), np.uint8))
					ifixpts_dilate = np.repeat(np.expand_dims(ifixpts_dilate, axis=2), 3, axis=2)
					iovermap[ifixpts_dilate>0.5] = 1

				iovermap = iovermap / np.max(iovermap) *255
				VideoWriter.write(im2uint8(iovermap))

			VideoCap.release()
			VideoWriter.release()



if __name__ == "__main__":

	DataSet = 'DIEM20'
	RootDir = 'E:/Code/IIP_Saliency_Video/DataSet/' + DataSet + '/'
	MethodNames = ['zk-TwoS']

	WITH_FIX = 1
	WITH_COLOT = 1
	visual_vid(RootDir, DataSet, MethodNames, with_color=WITH_COLOT, with_fix=WITH_FIX)
