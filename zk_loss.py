from __future__ import print_function

from keras import backend as K
from zk_config import *
import numpy as np


EPS = 2.2204e-16

def get_sum(input):
    return K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(input, axis=[1, 2])), shape_r_out, axis=1)), shape_c_out, axis=2)

def get_sum_3d(input):
    return K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(input, axis=[2, 3])), shape_r_out, axis=2)), shape_c_out, axis=3)

def get_max(input):
    return K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(input, axis=[1, 2])), shape_r_out, axis=1)), shape_c_out, axis=2)

def get_max_3d(input):
    return K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(input, axis=[2, 3])), shape_r_out, axis=2)), shape_c_out, axis=3)

def get_min(input):
    return K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.min(input, axis=[1, 2])), shape_r_out, axis=1)), shape_c_out, axis=2)

def get_min_3d(input):
    return K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.min(input, axis=[2, 3])), shape_r_out, axis=2)), shape_c_out, axis=3)

def get_mean(input):
    return K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.mean(input, axis=[1, 2])), shape_r_out, axis=1)), shape_c_out, axis=2)

def get_mean_3d(input):
    return K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.mean(input, axis=[2, 3])), shape_r_out, axis=2)), shape_c_out, axis=3)

def get_std(input):
    return K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.std(input, axis=[1, 2])), shape_r_out, axis=1)), shape_c_out, axis=2)

def get_std_3d(input):
    return K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.std(input, axis=[2, 3])), shape_r_out, axis=2)), shape_c_out, axis=3)

########################################################
# Loss function
# shape_r_out,shape_c_out = input.shape[1:3]
########################################################
# KL-Divergence Loss
def loss_kl(y_true, y_pred):
    y_true.set_shape(y_pred.shape)
    y_true /= (get_sum(y_true) + EPS)
    y_pred /= (get_sum(y_pred) + EPS)
    return K.sum(y_true * K.log((y_true / (y_pred + EPS)) + EPS), axis=[1,2])

def loss_kl_3d(y_true, y_pred):
    y_true.set_shape(y_pred.shape)
    y_true /= (get_sum_3d(y_true) + EPS)
    y_pred /= (get_sum_3d(y_pred) + EPS)
    return K.mean(K.sum(y_true * K.log((y_true / (y_pred + EPS)) + EPS), axis=[2,3]),axis=1)

# Correlation Coefficient Loss
def loss_cc(y_true, y_pred):
    y_true.set_shape(y_pred.shape)
    # y_true = (y_true - get_mean(y_true)) / get_std(y_true)
    # y_pred = (y_pred - get_mean(y_pred)) / get_std(y_pred)
    #
    # y_true = y_true - get_mean(y_true)
    # y_pred = y_pred - get_mean(y_pred)
    # r1 = K.sum(y_true * y_pred,axis=[1,2])
    # r2 = K.sqrt(K.sum(K.square(y_pred),axis=[1,2])*K.sum(K.square(y_true),axis=[1,2]))
    # return -2 * r1 / r2

    y_true /= (get_sum(y_true) + EPS)
    y_pred /= (get_sum(y_pred) + EPS)

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(y_true * y_pred, axis=[1,2])
    sum_x = K.sum(y_true, axis=[1,2])
    sum_y = K.sum(y_pred, axis=[1,2])
    sum_x_square = K.sum(K.square(y_true), axis=[1,2])
    sum_y_square = K.sum(K.square(y_pred), axis=[1,2])

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))
    return num / den

def loss_cc_3d(y_true, y_pred):
    y_true.set_shape(y_pred.shape)
    y_true /= (get_sum_3d(y_true) + EPS)
    y_pred /= (get_sum_3d(y_pred) + EPS)

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(y_true * y_pred, axis=[2,3])
    sum_x = K.sum(y_true, axis=[2,3])
    sum_y = K.sum(y_pred, axis=[2,3])
    sum_x_square = K.sum(K.square(y_true), axis=[2,3])
    sum_y_square = K.sum(K.square(y_pred), axis=[2,3])

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))
    return K.mean(num / den, axis=1)

# Normalized Scanpath Saliency Loss
def loss_nss(y_true, y_pred):
    y_pred = (y_pred - get_mean(y_pred)) / (get_std(y_pred)+ EPS)
    return K.sum(y_true * y_pred, axis=[1, 2]) / (K.sum(y_true, axis=[1, 2])+EPS)

def loss_nss_3d(y_true, y_pred):
    y_pred = (y_pred - get_mean_3d(y_pred)) / (get_std_3d(y_pred)+ EPS)
    return K.mean((K.sum(y_true * y_pred, axis=[2, 3]) / K.sum(y_true, axis=[2, 3])),axis=1)

def loss_sim_3d(y_true, y_pred):
    y_true.set_shape(y_pred.shape)
    y_true = (y_true - get_min_3d(y_true)) / (get_max_3d(y_true) - get_min_3d(y_true) + EPS)
    y_pred = (y_pred - get_min_3d(y_pred)) / (get_max_3d(y_pred) - get_min_3d(y_pred) + EPS)

    y_true /= (get_sum_3d(y_true) + EPS)
    y_pred /= (get_sum_3d(y_pred) + EPS)

    diff = K.minimum(y_true,y_pred)
    score = K.mean(K.sum(diff, axis=[2, 3]), axis=1)

    return score

def loss_sim(y_true, y_pred):
    y_true.set_shape(y_pred.shape)
    y_true = (y_true - get_min(y_true)) / (get_max(y_true) - get_min(y_true) + EPS)
    y_pred = (y_pred - get_min(y_pred)) / (get_max(y_pred) - get_min(y_pred) + EPS)

    y_true /= (get_sum(y_true) + EPS)
    y_pred /= (get_sum(y_pred) + EPS)

    diff = K.minimum(y_true,y_pred)
    score = K.sum(diff,axis=[1,2])

    return score

def loss_funet(y_true, y_pred):

    y_true_map = y_true[:,:,:,0:1]
    y_true_fix = y_true[:,:,:,1:2]

    # kl_loss = loss_kl(y_true_map, y_pred)
    # cc_loss = loss_cc(y_true_map, y_pred)
    # nss_loss = loss_nss(y_true_fix, y_pred)
    # return kl_loss + cc_loss + nss_loss

    # for kl loss
    y_true_map.set_shape(y_pred.shape)
    norm_y_true_map = y_true_map / (get_sum(y_true_map) + EPS)
    norm_y_pred     = y_pred / (get_sum(y_pred) + EPS)
    kl_loss = K.sum(norm_y_true_map * K.log((norm_y_true_map / (norm_y_pred + EPS)) + EPS), axis=[1, 2])

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(norm_y_true_map * norm_y_pred, axis=[1,2])
    sum_x = K.sum(norm_y_true_map, axis=[1,2])
    sum_y = K.sum(norm_y_pred, axis=[1,2])
    sum_x_square = K.sum(K.square(norm_y_true_map), axis=[1,2])
    sum_y_square = K.sum(K.square(norm_y_pred), axis=[1,2])

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))
    cc_loss = num / den

    # for nss loss
    y_pred_sal = (y_pred - get_mean(y_pred)) / (get_std(y_pred)+ EPS)
    nss_loss = K.sum(y_true_fix * y_pred_sal, axis=[1, 2]) / K.sum(y_true_fix, axis=[1, 2])

    return 10 * kl_loss - 2 * cc_loss - nss_loss

def loss_funet_3d(y_true, y_pred):

    y_true_map = y_true[:,:,:,:,0:1]
    y_true_fix = y_true[:,:,:,:,1:2]

    # kl_loss = loss_kl_3d(y_true_map, y_pred)
    # cc_loss = loss_cc_3d(y_true_map, y_pred)
    # nss_loss = loss_nss_3d(y_true_fix, y_pred)
    # return kl_loss + cc_loss + nss_loss

    # for kl loss
    y_true_map.set_shape(y_pred.shape)
    norm_y_true_map = y_true_map / (get_sum_3d(y_true_map) + EPS)
    norm_y_pred     = y_pred / (get_sum_3d(y_pred) + EPS)
    kl_loss = K.mean(K.sum(norm_y_true_map * K.log((norm_y_true_map / (norm_y_pred + EPS)) + EPS), axis=[2, 3]), axis=1)

    # for cc loss
    # y_true_map.set_shape(y_pred_sal.shape)
    # y_true_map /= (get_sum_3d(y_true_map) + EPS)
    # y_pred_sal /= (get_sum_3d(y_pred_sal) + EPS)

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(norm_y_true_map * norm_y_pred, axis=[2,3])
    sum_x = K.sum(norm_y_true_map, axis=[2,3])
    sum_y = K.sum(norm_y_pred, axis=[2,3])
    sum_x_square = K.sum(K.square(norm_y_true_map), axis=[2,3])
    sum_y_square = K.sum(K.square(norm_y_pred), axis=[2,3])

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))
    cc_loss = K.mean(num / den, axis=1)

    # for nss loss
    y_pred_sal = (y_pred - get_mean_3d(y_pred)) / (get_std_3d(y_pred) + EPS)
    nss_loss = K.mean((K.sum(y_true_fix * y_pred_sal, axis=[2, 3]) / K.sum(y_true_fix, axis=[2, 3])), axis=1)

    return 10 * kl_loss - 2 * cc_loss - nss_loss

