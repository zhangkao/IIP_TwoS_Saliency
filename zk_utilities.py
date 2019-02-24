from __future__ import division
import cv2
import numpy as np
import scipy.io
import scipy.ndimage
import hdf5storage as h5io
EPS = 2.2204e-16


#####################################################################
#Preprocess input and output video data
#####################################################################
def preprocess_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3),np.float32)

    for i, path in enumerate(paths):
        original_image = cv2.imread(path)
        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image

    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68

    return ims

def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1),np.float32)

    for i, path in enumerate(paths):
        original_map = cv2.imread(path, 0)
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i,:,:,0] = padded_map.astype(np.float32)
        ims[i,:,:,0] /= 255.0

    return ims

def preprocess_fixmaps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 1),np.uint8)

    for i, path in enumerate(paths):
        fix_map = scipy.io.loadmat(path)["I"]
        ims[i,:,:,0] = padding_fixation(fix_map, shape_r=shape_r, shape_c=shape_c)

    return ims

def preprocess_vidmaps(path, shape_r, shape_c, frames=float('inf')):

    fixmaps = h5io.loadmat(path)["fixMap"]
    h,w,c,nframes = fixmaps.shape
    nframes = min(nframes, frames)

    ims = np.zeros((nframes, shape_r, shape_c, 1),np.uint8)
    for i in range(nframes):
        original_map = fixmaps[:,:,:,i]
        ims[i, :, :, 0] = padding(original_map, shape_r, shape_c, 1)

    return ims

def preprocess_vidfixs(path, shape_r, shape_c, frames=float('inf')):

    fixmaps = h5io.loadmat(path)["fixLoc"]
    h,w,c,nframes = fixmaps.shape
    nframes = min(nframes, frames)

    ims = np.zeros((nframes, shape_r, shape_c, 1),np.uint8)
    for i in range(nframes):
        original_map = fixmaps[:,:,0,i]
        ims[i, :, :, 0] = padding_fixation(original_map, shape_r, shape_c)

    return ims

def preprocess_videos(path, shape_r, shape_c, frames=float('inf'), submean=False):

    cap = cv2.VideoCapture(path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    nframes = min(nframes,frames)
    ims = np.zeros((nframes, shape_r, shape_c, 3),np.uint8)
    for idx_frame in range(nframes):
        ret, frame = cap.read()
        # gray_im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame[:, :, 0] = gray_im
        # frame[:, :, 1] = gray_im
        # frame[:, :, 2] = gray_im
        ims[idx_frame] = padding(frame, shape_r, shape_c, 3)

    if submean:
        ims = ims.astype(np.float32)
        ims[:, :, :, 0] -= 103.939
        ims[:, :, :, 1] -= 116.779
        ims[:, :, :, 2] -= 123.68

    cap.release()

    return ims,nframes,height,width

def postprocess_predictions(pred, shape_r, shape_c):
    predictions_shape = pred.shape
    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[:, ((pred.shape[1] - shape_c) // 2):((pred.shape[1] - shape_c) // 2 + shape_c)]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[((pred.shape[0] - shape_r) // 2):((pred.shape[0] - shape_r) // 2 + shape_r), :]

    return img / np.max(img) * 255


def padding(img, shape_r=480, shape_c=640, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

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

def padding_fixation(img, shape_r=480, shape_c=640):
    img_padded = np.zeros((shape_r, shape_c),np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = resize_fixation(img, rows=shape_r, cols=new_cols)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_c)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def imgs_submean(input_imgs, mb = 103.939, mg = 116.779, mr = 123.68):
    imgs = np.zeros(input_imgs.shape,dtype=np.float32)
    if imgs.shape[3] == 3:
        imgs[:, :, :, 0] = input_imgs[:, :, :, 0] - mb
        imgs[:, :, :, 1] = input_imgs[:, :, :, 1] - mg
        imgs[:, :, :, 2] = input_imgs[:, :, :, 2] - mr
    else:
        raise NotImplementedError
    return imgs

def im2uint8(img):
    if img.dtype == np.uint8:
        return img
    else:
        img[img < 0] = 0
        img[img > 255] = 255
        img = np.rint(img).astype(np.uint8)
        return img

def np2mat(img, dtype=np.uint8):

    if dtype == np.uint8:
        return im2uint8(img)
    else:
        return img.astype(dtype)

#####################################################################
#Generate gaussmaps
#####################################################################
def st_get_gaussmaps(height,width,nb_gaussian):
    e = height / width
    e1 = (1 - e) / 2
    e2 = e1 + e

    mu_x = np.repeat(0.5,nb_gaussian,0)
    mu_y = np.repeat(0.5,nb_gaussian,0)

    sigma_x = e*np.array(np.arange(1,9))/16
    sigma_y = sigma_x

    x_t = np.dot(np.ones((height, 1)), np.reshape(np.linspace(0.0, 1.0, width), (1, width)))
    y_t = np.dot(np.reshape(np.linspace(e1, e2, height), (height, 1)), np.ones((1, width)))

    x_t = np.repeat(np.expand_dims(x_t, axis=-1), nb_gaussian, axis=2)
    y_t = np.repeat(np.expand_dims(y_t, axis=-1), nb_gaussian, axis=2)

    gaussian = 1 / (2 * np.pi * sigma_x * sigma_y + EPS) * \
               np.exp(-((x_t - mu_x) ** 2 / (2 * sigma_x ** 2 + EPS) +
                       (y_t - mu_y) ** 2 / (2 * sigma_y ** 2 + EPS)))

    return gaussian

def dy_get_gaussmaps(height,width,nb_gaussian):
    e = height / width
    e1 = (1 - e) / 2
    e2 = e1 + e

    mu_x = np.repeat(0.5,nb_gaussian,0)
    mu_y = np.repeat(0.5,nb_gaussian,0)


    sigma_x = np.array([1/4,1/4,1/4,1/4,
                        1/2,1/2,1/2,1/2])
    sigma_y = e*np.array([1 / 16, 1 / 8, 3 / 16, 1 / 4,
                          1 / 8, 1 / 4, 3 / 8, 1 / 2])

    x_t = np.dot(np.ones((height, 1)), np.reshape(np.linspace(0.0, 1.0, width), (1, width)))
    y_t = np.dot(np.reshape(np.linspace(e1, e2, height), (height, 1)), np.ones((1, width)))

    x_t = np.repeat(np.expand_dims(x_t, axis=-1), nb_gaussian, axis=2)
    y_t = np.repeat(np.expand_dims(y_t, axis=-1), nb_gaussian, axis=2)

    gaussian = 1 / (2 * np.pi * sigma_x * sigma_y + EPS) * \
               np.exp(-((x_t - mu_x) ** 2 / (2 * sigma_x ** 2 + EPS) +
                       (y_t - mu_y) ** 2 / (2 * sigma_y ** 2 + EPS)))

    return gaussian

def get_guasspriors(type='st', b_s=2, shape_r=60, shape_c=80, channels = 8):

    if type == 'dy':
        ims = dy_get_gaussmaps(shape_r, shape_c, channels)
    else:
        ims = st_get_gaussmaps(shape_r, shape_c, channels)

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims,b_s,axis=0)

    return ims

def get_guasspriors_3d(type = 'st', b_s = 2, time_dims=7,shape_r=60, shape_c=80, channels = 8):

    if type == 'dy':
        ims = dy_get_gaussmaps(shape_r, shape_c, channels)
    else:
        ims = st_get_gaussmaps(shape_r, shape_c, channels)

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims, time_dims, axis=0)

    ims = np.expand_dims(ims, axis=0)
    ims = np.repeat(ims, b_s, axis=0)
    return ims

