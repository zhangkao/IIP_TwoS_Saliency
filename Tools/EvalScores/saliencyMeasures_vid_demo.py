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
import scipy.io as sio

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

def getShufmap_vid(fixsDir, DataSet='DIEM20', size=None, maxframes = float('inf')):

	DataSet = DataSet.upper()
	if size is None:
		if DataSet in shuff_size.keys():
			size = shuff_size[DataSet]
		else:
			size = shuff_size["default"]

	fix_names = [f for f in os.listdir(fixsDir) if f.endswith('.mat')]
	fix_names.sort()
	fix_num = len(fix_names)

	# if DataSet == 'CITIUS':
	# 	fix_num = 45

	if DataSet == 'DIEM20':
		maxframes = 300

	ShufMap = np.zeros(size)
	for idx_n in range(fix_num):

		fixpts = h5io.loadmat(fixsDir + fix_names[idx_n])["fixLoc"]
		useframes = min(maxframes, fixpts.shape[3])
		fixpts = fixpts[:,:,:,:useframes]

		if fixpts.shape[:2] != size:
			# fixpts = np.array([resize_fixation(fixpts[:,:,0,i],size[0],size[1]) for i in range(useframes)]).transpose((1,2,0))
			fixpts = np.array([cv2.resize(fixpts[:, :, 0, i], (size[1], size[0]),interpolation=cv2.INTER_NEAREST) for i in range(useframes)]).transpose((1, 2, 0))
			fixpts = np.expand_dims(fixpts,axis=2)

		ShufMap += np.sum(fixpts[:,:,0,:],axis=2)

	h5io.savemat('Shuffle_' + DataSet + '3.mat', {'ShufMap': ShufMap})
	return ShufMap

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

def evalscores_vid(RootDir, DataSet, MethodNames, keys_order):

	mapsDir = RootDir + 'maps/'
	fixsDir = RootDir + 'fixations/maps/'

	salsDir = RootDir + 'Results/Saliency/'
	scoreDir = RootDir + 'Results/Scores_py/'
	if not os.path.exists(scoreDir):
		os.makedirs(scoreDir)

	print('Evaluate Metrics: ' + str(keys_order))
	if 'AUC_shuffled' in keys_order:
		if not os.path.exists(scoreDir):
			shuffle_map = getShufmap_vid(fixsDir,DataSet)
		else:
			shuffle_map = h5io.loadmat('Shuffle_' + DataSet + '.mat')["ShufMap"]
	else:
		shuffle_map = None

	for idx_m in range(len(MethodNames)):
		print("---" + str(idx_m + 1) + "/" + str(len(MethodNames)) + "---: " + MethodNames[idx_m])

		score_path = scoreDir + 'Score_' +MethodNames[idx_m] + '.mat'
		if os.path.exists(score_path):
			continue

		iscoreDir = scoreDir + MethodNames[idx_m] + '/'
		if not os.path.exists(iscoreDir):
			os.makedirs(iscoreDir)

		salmap_dir = salsDir + MethodNames[idx_m] + '/'
		sal_names = [f for f in os.listdir(salmap_dir) if f.endswith('.mat')]
		sal_names.sort()

		scores = {}
		for idx_n in range(len(sal_names)):
			print(str(idx_n + 1) + "/" + str(len(sal_names)) + ": " + sal_names[idx_n])

			file_name = sal_names[idx_n][:-4]
			iscore_path = iscoreDir + 'Score_' + file_name + '.mat'
			if os.path.exists(iscore_path):
				iscores = h5io.loadmat(iscore_path)["iscores"]
				scores[file_name] = iscores
				continue

			salmap = np.rint(h5io.loadmat(salmap_dir + file_name + '.mat')["salmap"]).astype(np.uint8)
			fixmap = h5io.loadmat(mapsDir + file_name + '_fixMaps.mat')["fixMap"]
			fixpts = h5io.loadmat(fixsDir + file_name + '_fixPts.mat')["fixLoc"]

			if shuffle_map.shape != fixpts.shape[:2]:
				# ishuffle_map = resize(shuffle_map, fixpts.shape, order=3, mode='nearest')
				ishuffle_map = resize_fixation(shuffle_map, fixpts.shape[0], fixpts.shape[1])

			else:
				ishuffle_map = shuffle_map

			nframes = min(salmap.shape[3],min(fixpts.shape[3],fixmap.shape[3]))
			iscores = np.zeros((nframes, len(keys_order)))
			for idx_f in range(nframes):
				isalmap = salmap[:, :, 0, idx_f]/255.0
				ifixmap = fixmap[:, :, 0, idx_f]/255.0
				ifixpts = fixpts[:, :, 0, idx_f]
				if not np.any(isalmap) or not np.any(ifixmap) or not np.any(ifixpts):
					iscores[idx_f] = np.NaN
					print(str(idx_f + 1) + "/" + str(nframes) + ": failed!")
					continue

				values = getSimVal(isalmap, ifixmap, ifixpts, ishuffle_map)
				iscores[idx_f] = values
				# print(str(idx_f + 1) + "/" + str(nframes) + ": finished!")

			scores[file_name] = iscores
			h5io.savemat(iscore_path, {'iscores': iscores})

		h5io.savemat(score_path, {'scores': scores})


def getAllScores_mean(RootDir, MaxVidNums=float('inf')):

	scoreDir = RootDir + 'Results/Scores_py/'

	score_names = [f for f in os.listdir(scoreDir) if f.endswith('.mat')]
	score_names.sort()
	method_num = len(score_names)

	meanS = {}
	for idx_m in range(method_num):
		method_name = score_names[idx_m]
		iscores = h5io.loadmat(scoreDir + method_name)["scores"]
		vid_num = min(len(iscores), MaxVidNums)

		iscores_mean_vid = [np.mean(v, 0) for i, v in zip(range(len(iscores)), iscores.values()) if i < vid_num]
		iscores_mean_vid = np.array(iscores_mean_vid)
		iscores_mean = np.mean(iscores_mean_vid, 0)

		tmp_name = method_name[6:-4].replace('-','_')
		meanS[tmp_name] = {'meanS': iscores_mean, 'meanS_vid': iscores_mean_vid, 'scores': iscores}

	h5io.savemat(RootDir + 'Results/meanS_py.mat', {'meanS': meanS})



if __name__ == "__main__":

	DataSet = 'citius'
	RootDir = '/home/zk/zk/TwoS-release/DataSet/' + DataSet + '/'
	keys_order = ['AUC_shuffled', 'NSS', 'AUC_Judd', 'AUC_Borji', 'KLD', 'SIM', 'CC']
	MethodNames = ['ACL']

	IS_EVAL_SCORES=1
	if IS_EVAL_SCORES:
		evalscores_vid(RootDir, DataSet, MethodNames, keys_order)

	IS_ALL_SCORES = 1
	if IS_ALL_SCORES:
		MaxVideoNums = float('inf')
		if DataSet.upper() == 'CITIUS':
			MaxVideoNums = 45

		getAllScores_mean(RootDir , MaxVideoNums)
