from __future__ import division

import os, cv2, sys
import numpy as np

import keras.backend as K
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from zk_config import *
from zk_utilities import *
from zk_models import *
from zk_loss import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def generator(b_s, phase_gen='train'):
    if phase_gen == 'train':
        images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [fixs_train_path + f for f in os.listdir(fixs_train_path) if f.endswith('.mat')]
    elif phase_gen == 'val':
        images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [fixs_val_path + f for f in os.listdir(fixs_val_path) if f.endswith('.mat')]
    else:
        raise NotImplementedError

    images.sort()
    maps.sort()
    fixs.sort()

    counter = 0
    while True:
        X_img = preprocess_images(images[counter:counter + b_s], shape_r, shape_c)
        Y_map = preprocess_maps(maps[counter:counter+b_s], shape_r_out, shape_c_out)
        Y_fix = preprocess_fixmaps(fixs[counter:counter + b_s], shape_r_out, shape_c_out)

        X_cb = get_guasspriors('st', b_s, shape_r_out, shape_c_out, nb_gaussian)

        yield [X_img, X_cb], [Y_map, Y_map, Y_fix]
        counter = (counter + b_s) % len(images)

def generator_test(b_s, imgs_test_path):
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if (f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png'))]
    images.sort()

    counter = 0
    while True:
        X_img = preprocess_images(images[counter:counter + b_s], shape_r, shape_c)
        X_cb = get_guasspriors('st',b_s, shape_r_out, shape_c_out, nb_gaussian)
        yield [X_img, X_cb]
        counter = (counter + b_s) % len(images)


if __name__ == '__main__':

    phase = 'train'
    method_name = 'zk-twos-st'

    print("Build Static SalCNN Model: " )
    model = salcnn_Static_Net(img_cols=shape_c, img_rows=shape_r, img_channels=3)
    sgd = SGD(lr=1e-3, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile( optimizer=sgd, loss=[loss_kl, loss_cc, loss_nss], loss_weights=[10,-2,-1])

    if phase == 'train':

        tmdir = 'Models/temp_models/' + method_name
        tmp_model_dir = tmdir + '/' + method_name + '_'
        # logdir = tmdir +'/log/'
        if not os.path.exists(tmdir):
            os.makedirs(tmdir)

        pre_model = 'Models/zk-twos-st-pre-model.h5'
        if os.path.exists(pre_model):
            print("Load weights SalCNN")
            model.load_weights(pre_model, by_name=True)

        print("Training SalCNN")
        model.fit_generator(generator(b_s=bs_st_c2d), steps_per_epoch=math.ceil(nb_imgs_train / bs_st_c2d),
                                epochs=epochs, validation_data=generator(b_s=bs_st_c2d, phase_gen='val'),
                                validation_steps=math.ceil(nb_imgs_val / bs_st_c2d),
                                callbacks=[EarlyStopping(patience=5), ModelCheckpoint(tmp_model_dir + '{epoch:02d}_{val_loss:.4f}.h5', save_best_only=False)])

    elif phase == "test":

        dataset       = 'salicon/val'
        model_path    = 'Models/zk-twos-st-model.h5'

        output_folder = dataDir + '/' + dataset + '/Results/Saliency/' + method_name + '/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        imgs_test_path = dataDir + '/' + dataset + '/images/'
        file_names = [f for f in os.listdir(imgs_test_path) if (f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png'))]
        file_names.sort()
        nb_imgs_test = len(file_names)

        print("Load weights SalCNN")
        model.load_weights(model_path)

        print("Predict saliency maps for " + imgs_test_path)
        predictions = model.predict_generator(generator_test(b_s=bs_st_c2d, imgs_test_path=imgs_test_path), math.ceil(nb_imgs_test / bs_st_c2d))

        for pred, name in zip(predictions[0], file_names):
            original_image = cv2.imread(imgs_test_path + name, 0)
            predimg = pred[:, :, 0]

            res = postprocess_predictions(predimg, original_image.shape[0], original_image.shape[1])
            cv2.imwrite(output_folder + '%s' % name[:-3] + 'png', res.astype(int))
    
    else:
        raise NotImplementedError
