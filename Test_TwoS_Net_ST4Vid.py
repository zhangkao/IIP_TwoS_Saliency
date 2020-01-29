from __future__ import division
import os, cv2, sys, math,random
import numpy as np
import hdf5storage as h5io
import keras.backend as K

from zk_config import *
from zk_utilities import *
from zk_models import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    camera_number = 0
    cap = cv2.VideoCapture(camera_number)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    IS_SAVE_RESULTS = 0
    if IS_SAVE_RESULTS:
        output_name = 'result.mp4'
        output_path = 'Results/Saliency/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        VideoWriter = cv2.VideoWriter(output_path+output_name, fourcc, 20, (width, height), isColor=True)

    print("Build SalCNN Model: " )
    model_path = 'Models/zk-twos-st-model.h5'
    model = salcnn_Static_Net(img_cols=shape_c, img_rows=shape_r, img_channels=3)
    model.load_weights(model_path)

    X_cb = get_guasspriors('st', 1, shape_r_out, shape_c_out, 8)
    while (True):
        ret, frame = cap.read()

        ims = cv2.resize(frame,(shape_c,shape_r))
        # ims = padding(frame, shape_r, shape_c, 3)

        ims = ims.astype(np.float32)
        ims[:, :, 0] -= 103.939
        ims[:, :, 1] -= 116.779
        ims[:, :, 2] -= 123.68

        X_img = np.expand_dims(ims,0)
        prediction = model.predict([X_img, X_cb],1)

        salmap = cv2.resize(prediction[0][0,:,:,0], (width, height))
        # out = postprocess_predictions(prediction,height, width)

        overmap = heatmap_overlay(frame, salmap)

        cv2.imshow('saliency', overmap)

        if IS_SAVE_RESULTS:
            overmap = overmap / np.max(overmap) * 255
            VideoWriter.write(im2uint8(overmap))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if IS_SAVE_RESULTS:
        VideoWriter.release()

