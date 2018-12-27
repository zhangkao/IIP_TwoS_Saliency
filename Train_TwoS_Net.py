from __future__ import division
import os, cv2, sys
import numpy as np
import hdf5storage as h5io
import math, random

from keras.optimizers import SGD
import keras.backend as K

from zk_config import *
from zk_utilities import *
from zk_models import *
from zk_loss import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    method_name = 'zk_TwoS'
    tmdir = '/Models/temp_models/' + method_name
    tmp_model_path = tmdir + '/' + method_name + '_'
    if not os.path.exists(tmdir):
        os.makedirs(tmdir)

    st_pre_model_path = '/Models/zk-st-VGG16-FPN-CGP.h5'
    twos_pre_model_path = '/Models/zk-twos-final-model.h5'

    print("Build SalCNN Model: ")
    fu_model = salcnn_TwoS_Net(time_dims=nb_c3dframes, img_cols=shape_c, img_rows=shape_r, img_channels=3, pre_sf_path=st_pre_model_path)

    sgd = SGD(lr=1e-3, decay=0.0005, momentum=0.9, nesterov=True)
    fu_model.compile(optimizer=sgd, loss=[loss_kl_3d, loss_cc_3d, loss_nss_3d], loss_weights=[10,-2,-1])

    if os.path.exists(twos_pre_model_path):
        print("Load pre-train pre-train weights")
        fu_model.load_weights(twos_pre_model_path)

    print("Training SalCNN Model")
    IS_EARLY_STOP = True
    if IS_EARLY_STOP:
        min_delta = 0.00001
        patience = 4

        val_patience = 0
        min_val_loss = 10000
        mean_val_loss = 0

    X_cb_st = get_guasspriors_3d('st',bs_dy_c3d, nb_c3dframes, shape_r_out, shape_c_out, nb_gaussian)
    X_cb_dy = get_guasspriors_3d('dy', bs_dy_c3d, nb_c3dframes, shape_r_out, shape_c_out, nb_gaussian)
    for epoch in range(epochs):
        print("epochs: [%d / %d] " % (epoch + 1, epochs))

        ###########################
        # train
        ###########################
        videos_list, vidmaps_list, vidfixs_list = read_video_list('train',shuffle=True)
        num_step = 0
        train_loss = 0.0
        for idx_video in range(len(videos_list)):
            print("videos: %d/%d, train with data from :%s" % (idx_video+1,len(videos_list), videos_list[idx_video]))

            vidmaps = preprocess_vidmaps(vidmaps_list[idx_video], shape_r_out, shape_c_out)
            vidfixs = preprocess_vidfixs(vidfixs_list[idx_video], shape_r_out, shape_c_out)

            vidimgs, nframes, height, width = preprocess_videos(videos_list[idx_video], shape_r, shape_c)
            nframes = min(min(vidfixs.shape[0],vidmaps.shape[0]),nframes)

            count_bs = int(nframes / nb_c3dframes)
            saveframes = count_bs * nb_c3dframes
            vidimgs = vidimgs[0:saveframes].reshape((count_bs, nb_c3dframes, shape_r, shape_c, 3))
            vidmaps = vidmaps[0:saveframes].reshape((count_bs, nb_c3dframes, shape_r_out, shape_c_out, 1))
            vidfixs = vidfixs[0:saveframes].reshape((count_bs, nb_c3dframes, shape_r_out, shape_c_out, 1))

            bs_steps = math.ceil(count_bs/bs_dy_c3d)
            video_loss = 0.0
            for idx_bs in range(bs_steps):
                x_imgs = vidimgs[idx_bs * bs_dy_c3d:(idx_bs + 1) * bs_dy_c3d]
                y_maps = vidmaps[idx_bs * bs_dy_c3d:(idx_bs + 1) * bs_dy_c3d]
                y_fixs = vidfixs[idx_bs * bs_dy_c3d:(idx_bs + 1) * bs_dy_c3d]

                if not np.any(y_fixs,axis=(2,3)).all():
                    continue

                if x_imgs.shape[0] != bs_dy_c3d:
                    tX_cb_st = get_guasspriors_3d('st', x_imgs.shape[0], nb_c3dframes, shape_r_out, shape_c_out, nb_gaussian)
                    tX_cb_dy = get_guasspriors_3d('dy', x_imgs.shape[0], nb_c3dframes, shape_r_out, shape_c_out, nb_gaussian)
                    X_input = [x_imgs, tX_cb_st, tX_cb_dy]
                else:
                    X_input = [x_imgs, X_cb_st, X_cb_dy]

                out_loss = fu_model.train_on_batch(X_input, [y_maps,y_maps,y_fixs])
                print("video: %d/%d, batch: %d/%d, train loss : %.4f  (Kl: %.4f, cc: %.4f, nss: %.4f)" % (idx_video+1, len(videos_list), idx_bs + 1, bs_steps, out_loss[0], out_loss[1], out_loss[2], out_loss[3]))

                video_loss += out_loss[0]
                train_loss += out_loss[0]
                num_step += 1
            print("videos: %d/%d, mean train loss: %.4f " % (idx_video+1, len(videos_list), video_loss / bs_steps))

        print("mean train loss: %.4f " % (train_loss / (num_step * 10)))

        ###########################
        # val
        ###########################
        videos_list, vidmaps_list, vidfixs_list = read_video_list('val', shuffle=False)
        num_step = 0
        val_loss = 0.0
        for idx_video in range(len(videos_list)):
            print("videos: %d/%d, val with data from :%s" % (idx_video+1, len(videos_list), videos_list[idx_video]))

            vidmaps = preprocess_vidmaps(vidmaps_list[idx_video], shape_r_out, shape_c_out)
            vidfixs = preprocess_vidfixs(vidfixs_list[idx_video], shape_r_out, shape_c_out)

            vidimgs, nframes, height, width = preprocess_videos(videos_list[idx_video], shape_r, shape_c)
            nframes = min(min(vidfixs.shape[0],vidmaps.shape[0]),nframes)

            count_bs = int(nframes / nb_c3dframes)
            saveframes = count_bs * nb_c3dframes
            vidimgs = vidimgs[0:saveframes].reshape((count_bs, nb_c3dframes, shape_r, shape_c, 3))
            vidmaps = vidmaps[0:saveframes].reshape((count_bs, nb_c3dframes, shape_r_out, shape_c_out, 1))
            vidfixs = vidfixs[0:saveframes].reshape((count_bs, nb_c3dframes, shape_r_out, shape_c_out, 1))

            bs_steps = math.ceil(count_bs/bs_dy_c3d)
            video_loss = 0.0
            for idx_bs in range(bs_steps):
                x_imgs = vidimgs[idx_bs * bs_dy_c3d:(idx_bs + 1) * bs_dy_c3d]
                y_maps = vidmaps[idx_bs * bs_dy_c3d:(idx_bs + 1) * bs_dy_c3d]
                y_fixs = vidfixs[idx_bs * bs_dy_c3d:(idx_bs + 1) * bs_dy_c3d]

                if not np.any(y_fixs,axis=(2,3)).all():
                    continue

                if x_imgs.shape[0] != bs_dy_c3d:
                    tX_cb_st = get_guasspriors_3d('st', x_imgs.shape[0], nb_c3dframes, shape_r_out, shape_c_out, nb_gaussian)
                    tX_cb_dy = get_guasspriors_3d('dy', x_imgs.shape[0], nb_c3dframes, shape_r_out, shape_c_out, nb_gaussian)
                    X_input = [x_imgs, tX_cb_st, tX_cb_dy]
                else:
                    X_input = [x_imgs, X_cb_st, X_cb_dy]

                out_loss = fu_model.test_on_batch(X_input, [y_maps,y_maps,y_fixs])
                print("video: %d/%d, batch: %d/%d, val loss : %.4f  (Kl: %.4f, cc: %.4f, nss: %.4f)" % (idx_video+1, len(videos_list), idx_bs + 1, bs_steps, out_loss[0], out_loss[1], out_loss[2], out_loss[3]))

                video_loss += out_loss[0]
                val_loss += out_loss[0]
                num_step += 1
            print("video : %d/%d, mean val loss: %.4f " % (idx_video+1, len(videos_list), video_loss / bs_steps))

        front_val_loss = mean_val_loss
        mean_val_loss = val_loss/(num_step*10)
        print("mean val loss: %.4f " % (mean_val_loss))
        save_path = tmp_model_path + '_%02d_%.4f.h5' % (epoch,mean_val_loss)
        fu_model.save(save_path)

        ###########################
        # eary stop
        ###########################
        if IS_EARLY_STOP:
            #patience > num
            if mean_val_loss < min_val_loss:
                min_val_loss = mean_val_loss
                min_val_epoch = epoch
                val_patience = 0
            else:
                val_patience +=1
                if val_patience > patience:
                    print('Early stop')
                    break
            # min_delta < num
            if abs(front_val_loss - mean_val_loss) < min_delta:
                print('Early stop')
                break
