#! /usr/bin/env python3
'''
Implemented by Erwan DAVID (IPI, LS2N, Nantes, France), 2018

E. DAVID, J. Guttiérez, A Coutrot, M. Perreira Da Silva, P. Le Callet (...). A Dataset of Head and Eye Movements for 360° Videos. ACM MMSys18, dataset and toolbox track
'''

from functools import partial
import numpy as np
from numpy import random
from skimage import exposure
from skimage import img_as_float
from skimage.transform import resize
import matplotlib.pyplot as plt
import hdf5storage as h5io
import re, os, glob, cv2

EPSILON = np.finfo('float').eps

from metrics import *

metrics = {
	"AUC_shuffled": [AUC_shuffled, 'fix', True], # Binary fixation map
	"AUC_Judd": [AUC_Judd, 'fix', False], # Binary fixation map
	"AUC_Borji": [AUC_Borji, 'fix', False], # Binary fixation map
	"NSS": [NSS, 'fix', False], # Binary fixation map
	"CC": [CC, 'sal', False], # Saliency map
	"SIM": [SIM, 'sal', False], # Saliency map
	"KLD": [KLD, 'sal', False],  #  Saliency map
}

shuff_size = {
	"SALICON": (480,640),
	"DIEM":    (480,640),
	"DIEM20":  (480,640),
	"CITIUS":  (240,320),
	"SFU":     (288,352),
	"default": (480,640),
}

def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols),np.uint8)
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1

    return out

def getSimVal(salmap1, salmap2, fixation_map=None, othermap=None):
	values = []

	for metric in keys_order:

		func = metrics[metric][0]
		compType = metrics[metric][1]
		sim = metrics[metric][2]

		if compType == "fix":
			if sim and not type(None) in [type(fixation_map), type(othermap)]:
				m = func(salmap1, fixation_map, othermap)
			else:
				m = func(salmap1, fixation_map)
		else:
			m = func(salmap1, salmap2)
		values.append(m)
	return values

def evalscores_img(RootDir, DataSet, MethodNames, keys_order):

	mapsDir = RootDir + 'maps/'
	fixsDir = RootDir + 'fixations/maps/'

	salsDir = RootDir + 'Results/Saliency/'
	scoreDir = RootDir + 'Results/Scores_py/'
	if not os.path.exists(scoreDir):
		os.makedirs(scoreDir)

	print('Evaluate Metrics: ' + str(keys_order))
	if 'AUC_shuffled' in keys_order:
		if not os.path.exists(scoreDir):
			shuffle_map = getShufmap_img(fixsDir,DataSet)
		else:
			shuffle_map = h5io.loadmat('Shuffle_' + DataSet + '.mat')["ShufMap"]
	else:
		shuffle_map = None

	for idx_m in range(len(MethodNames)):
		print("---" + str(idx_m + 1) + "/" + str(len(MethodNames)) + "---: " + MethodNames[idx_m])

		score_path = scoreDir + 'Score_' +MethodNames[idx_m] + '.mat'
		if os.path.exists(score_path):
			continue

		salmap_dir = salsDir + MethodNames[idx_m] + '/'
		sal_names = [f for f in os.listdir(salmap_dir) if f.endswith('.png')]
		sal_names.sort()

		scores = np.zeros((len(sal_names),len(keys_order)))
		for idx_n in range(len(sal_names)):
			file_name = sal_names[idx_n]

			salmap = cv2.imread(salmap_dir + file_name,-1)/255.0
			fixmap = cv2.imread(mapsDir + file_name,-1)/255.0
			fixpts = h5io.loadmat(fixsDir + file_name[:-4] + '.mat')["I"]

			if shuffle_map.shape != fixpts.shape[:2]:
				# ishuffle_map = resize(shuffle_map, fixpts.shape, order=3, mode='nearest')
				ishuffle_map = resize_fixation(shuffle_map, fixpts.shape[0], fixpts.shape[1])

			else:
				ishuffle_map = shuffle_map

			if not np.any(salmap) or not np.any(fixmap) or not np.any(fixpts) or not np.any(ishuffle_map):
				scores[idx_n] = np.NaN
				print(str(idx_n) + "/" + str(len(sal_names)) + ": failed!")
				continue

			values = getSimVal(salmap, fixmap, fixpts, ishuffle_map)
			scores[idx_n] = values
			print(str(idx_n+1) + "/" + str(len(sal_names)) + ": finished!")

		h5io.savemat(score_path, {'scores': scores})

def getShufmap_img(fixsDir, DataSet='SALICON', size=None):

	DataSet = DataSet.upper()
	if size is None:
		if DataSet in shuff_size.keys():
			size = shuff_size[DataSet]
		else:
			size = shuff_size["default"]

	fix_names = [f for f in os.listdir(fixsDir) if f.endswith('.mat')]
	fix_names.sort()

	ShufMap = np.zeros(size)
	for idx_n in range(len(fix_names)):

		fixpts = h5io.loadmat(fixsDir + fix_names[idx_n])["I"] > 0.5
		if fixpts.shape != size:
			# fixpts = resize(fixpts, (size[0],size[1]), order=3, mode='nearest')
			fixpts = resize_fixation(fixpts,size[0],size[1])
		ShufMap += fixpts

	h5io.savemat('Shuffle_' + DataSet + '.mat', {'ShufMap': ShufMap})
	return ShufMap


def getAllScores_mean(RootDir, MaxImgNums=float('inf')):

	scoreDir = RootDir + 'Results/Scores_py/'

	score_names = [f for f in os.listdir(scoreDir) if f.endswith('.mat')]
	score_names.sort()
	method_num = len(score_names)

	meanS = {}
	for idx_m in range(method_num):
		method_name = score_names[idx_m]
		iscores = h5io.loadmat(scoreDir + method_name)["scores"]
		img_num = min(len(iscores), MaxImgNums)
		iscores_mean = np.mean(iscores[:img_num], 0)

		tmp_name = method_name[6:-4].replace('-','_')
		meanS[tmp_name] = {'meanS': iscores_mean,  'scores': iscores}

	h5io.savemat(RootDir + 'Results/meanS_py.mat', {'meanS': meanS})



if __name__ == "__main__":

	DataSet = 'salicon'
	RootDir = 'E:/Code/IIP_Saliency_Video/DataSet/salicon/val/'
	keys_order = ['AUC_shuffled', 'NSS', 'AUC_Judd', 'AUC_Borji', 'KLD', 'SIM', 'CC']
	MethodNames = ['zk-twos-st']

	IS_EVAL_SCORES=1
	if IS_EVAL_SCORES:
		evalscores_img(RootDir, DataSet, MethodNames, keys_order)

	IS_ALL_SCORES = 1
	if IS_ALL_SCORES:

		# for python implementation
		MaxVideoNums = float('inf')
		getAllScores_mean(RootDir , MaxVideoNums)

		# for matlab implementation
		# import matlab
		# import matlab.engine
		# eng = matlab.engine.start_matlab()
		# eng.Img_MeanScore(RootDir, nargout = 0)
		# eng.Image_Meanscore(nargout = 0)