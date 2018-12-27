from __future__ import division
import os, cv2, sys
import numpy as np
import hdf5storage as h5io
import math
import keras.backend as K

from zk_config import *
from zk_utilities import *
from zk_models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    dataset = 'DIEM20'
    IS_SAVE_SMALL = 0
    SaveFrames = float('inf')
    OutDtype = np.uint8

    if dataset == 'DIEM20':
        SaveFrames = 300

    method_name = 'zk-TwoS'
    model_path = 'Models/zk-twos-final-model.h5'

    input_path  = dataDir + '/' + dataset + '/Videos/'
    output_path = dataDir + '/' + dataset + '/Results/Saliency/' + method_name + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("Build SalCNN Model: " )
    fu_model = salcnn_TwoS_Net(time_dims=nb_c3dframes, img_cols=shape_c, img_rows=shape_r, img_channels=3, pre_sf_path='')
    fu_model.load_weights(model_path)

    file_names = [f for f in os.listdir(input_path) if (f.endswith('.avi') or f.endswith('.mp4'))]
    file_names.sort()
    nb_videos_test = len(file_names)

    for idx_video in range(nb_videos_test):
        print("%d/%d   " % (idx_video + 1, nb_videos_test) + file_names[idx_video])

        if IS_SAVE_SMALL:
            ovideo_path = output_path + (file_names[idx_video])[:-4] + '_60_80.mat'
        else:
            ovideo_path = output_path + (file_names[idx_video])[:-4] + '.mat'
        if os.path.exists(ovideo_path):
            continue

        ivideo_path = input_path + file_names[idx_video]
        imgs, nframes, height, width = preprocess_videos(ivideo_path, shape_r, shape_c, SaveFrames + nb_c3dframes - 1)

        count_bs = int(nframes / nb_c3dframes)
        isaveframes = count_bs * nb_c3dframes
        x_imgs = imgs[0:isaveframes].reshape((count_bs, nb_c3dframes, shape_r, shape_c, 3))

        X_cb_st = get_guasspriors_3d('st', count_bs, nb_c3dframes, shape_r_out, shape_c_out, nb_gaussian)
        X_cb_dy = get_guasspriors_3d('dy', count_bs, nb_c3dframes, shape_r_out, shape_c_out, nb_gaussian)
        X_input = [x_imgs, X_cb_st, X_cb_dy]

        predictions = fu_model.predict(X_input, bs_dy_c3d, verbose=0)

        predi = predictions[0].reshape((isaveframes, shape_r_out, shape_c_out, 1))
        if IS_SAVE_SMALL:
            pred_mat = np2mat(predi,dtype=OutDtype)
            # ovideo_path = output_path + (file_names[idx_video])[:-4] + '_60_80.mat'
            h5io.savemat(ovideo_path, {'salmap': pred_mat})
        else:
            pred_mat = np.zeros((isaveframes, height, width, 1),dtype=OutDtype)
            for idx_pre, ipred in zip(range(isaveframes), predi):
                isalmap = postprocess_predictions(ipred, height, width)
                pred_mat[idx_pre, :, :, 0] = np2mat(isalmap,dtype=OutDtype)

            iSaveFrame = min(isaveframes, SaveFrames)
            pred_mat = pred_mat[0:iSaveFrame, :, :, :].transpose((1, 2, 3, 0))
            # ovideo_path = output_path + (file_names[idx_video])[:-4] + '.mat'
            h5io.savemat(ovideo_path, {'salmap':pred_mat})

        print("finished ..")
    print("Done ..")
