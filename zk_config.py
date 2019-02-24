import math,random,os

#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# replace the datadir to your path
dataDir = 'E:/Code/IIP_Saliency_Video/DataSet'
#########################################################################
# Parameters SETTINGS
#########################################################################
# batch size
bs_st_c2d = 8
bs_dy_c3d = 1

# number of frames input conv3d or lstm
nb_c3dframes = 7
# number of rows of input images
shape_r = 480
# number of cols of input images
shape_c = 640
# number of rows of model outputs
shape_r_out = int(shape_r/8)
# number of cols of model outputs
shape_c_out = int(shape_c/8)
# number of epochs
epochs = 20
# number of learned priors
nb_gaussian = 8


#########################################################################
# Images TRAINING SETTINGS
#########################################################################
imgs_data_path = dataDir + '/salicon'
# path of training images
imgs_train_path = imgs_data_path + '/train/images/'
# path of training maps
maps_train_path = imgs_data_path + '/train/maps/'
# path of training fixation maps
fixs_train_path = imgs_data_path + '/train/fixations/maps/'
# number of training images
nb_imgs_train = 10000
# path of validation images
imgs_val_path = imgs_data_path + '/val/images/'
# path of validation maps
maps_val_path = imgs_data_path + '/val/maps/'
# path of validation fixation maps
fixs_val_path = imgs_data_path + '/val/fixations/maps/'
# number of validation images
nb_imgs_val = 5000

#########################################################################
# Videos TRAINING SETTINGS
#########################################################################
videos_data_path = dataDir + '/salVideo'
# path of training images
videos_train_path = videos_data_path + '/train/videos/'
# path of training maps
videomaps_train_path = videos_data_path + '/train/maps/'
# path of training fixation maps
videofixs_train_path = videos_data_path + '/train/fixations/maps/'
# path of training flows
videoflows_train_path = videos_data_path + '/train/flows/'
# number of training images
# nb_videos_train = 44

# path of validation images
videos_val_path = videos_data_path + '/val/videos/'
# path of validation maps
videomaps_val_path = videos_data_path + '/val/maps/'
# path of validation fixation maps
videofixs_val_path = videos_data_path + '/val/fixations/maps/'
# path of validation flows
videoflows_val_path = videos_data_path + '/val/flows/'
# number of validation images
# nb_videos_val = 20

video_train_list = videos_data_path + '/train.txt'
video_val_list   = videos_data_path + '/val.txt'

#########################################################################
# Read/Get Videos List
#########################################################################
def read_video_list(phase_gen='train', shuffle=True):
    if phase_gen == 'train':
        read_path = video_train_list
        videos_path = videos_train_path
        videomaps_path = videomaps_train_path
        videofixs_path = videofixs_train_path
    elif phase_gen == 'val':
        read_path = video_val_list
        videos_path = videos_val_path
        videomaps_path = videomaps_val_path
        videofixs_path = videofixs_val_path
    else:
        raise NotImplementedError

    f = open(read_path)
    lines = f.readlines()

    if shuffle:
        random.shuffle(lines)

    videos = [videos_path + f.strip('\n') + '.mp4' for f in lines]
    vidmaps = [videomaps_path + f.strip('\n') + '_fixMaps.mat' for f in lines]
    vidfixs = [videofixs_path + f.strip('\n') + '_fixPts.mat' for f in lines]
    f.close()

    return videos, vidmaps, vidfixs

def get_video_list(phase_gen='train', shuffle=True):
    if phase_gen == 'train':
        videos  = [videos_train_path + f for f in os.listdir(videos_train_path) if (f.endswith('.avi') or f.endswith('.mp4'))]
        vidmaps = [videomaps_train_path + f for f in os.listdir(videomaps_train_path) if f.endswith('.mat')]
        vidfixs = [videofixs_train_path + f for f in os.listdir(videofixs_train_path) if f.endswith('.mat')]
    elif phase_gen == 'val':
        videos  = [videos_val_path + f for f in os.listdir(videos_val_path) if (f.endswith('.avi') or f.endswith('.mp4'))]
        vidmaps = [videomaps_val_path + f for f in os.listdir(videomaps_val_path) if f.endswith('.mat')]
        vidfixs = [videofixs_val_path + f for f in os.listdir(videofixs_val_path) if f.endswith('.mat')]
    else:
        raise NotImplementedError

    if shuffle:
        out = list(zip(videos, vidmaps, vidfixs))
        random.shuffle(out)
        videos, vidmaps, vidfixs = zip(*out)
    else:
        videos.sort()
        vidmaps.sort()
        vidfixs.sort()

    return videos, vidmaps, vidfixs
