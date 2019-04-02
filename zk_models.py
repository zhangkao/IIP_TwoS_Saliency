from __future__ import print_function

from keras import layers
from keras.models import Model,load_model
from keras.layers import Input, Conv2D, MaxPooling2D,  Activation, Conv3D, TimeDistributed, UpSampling2D
from keras.initializers import Constant
from keras.applications.vgg16 import VGG16

from zk_config import *
from salcnn_vgg16 import *
EPS = 2.2204e-16

########################################################
# 1.0 SF-NET
########################################################
def salcnn_SF_Net(img_rows=480, img_cols=640, img_channels=3):

	sal_input = Input(shape=(img_rows, img_cols, img_channels))
	input_shape = (img_rows, img_cols, img_channels)

	# cnn = salcnn_VGG16(include_top=False, weights='imagenet', input_tensor=sal_input, input_shape=input_shape)
	cnn = VGG16(include_top=False, weights='imagenet', input_tensor=sal_input, input_shape=input_shape)

	# C2 = cnn.get_layer(name='block2_pool').output
	C3 = cnn.get_layer(name='block3_pool').output
	C4 = cnn.get_layer(name='block4_pool').output
	C5 = cnn.get_layer(name='block5_conv3').output

	# C2_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='sal_fpn_c2')(C2)
	C3_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='sal_fpn_c3')(C3)
	C4_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='sal_fpn_c4')(C4)
	C5_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='sal_fpn_c5')(C5)

	C5_1_up = UpSampling2D((2, 2), interpolation='bilinear', name='sal_fpn_p5_up')(C5_1)
	C4_1_up = UpSampling2D((2, 2), interpolation='bilinear', name='sal_fpn_p4_up')(C4_1)
	x = layers.concatenate([C3_1, C4_1_up, C5_1_up], axis=-1, name='sal_fpn_merge_concat')
	model = Model(inputs=[sal_input], outputs=[x], name='salcnn_sf_fpn')
	return model

#######################################################
# 2.0 TwoS-Net
#######################################################
def salcnn_Static_Net(img_rows=480, img_cols=640, img_channels=3, pre_sf_path = ''):

	sfnet = salcnn_SF_Net(img_rows=img_rows, img_cols=img_cols, img_channels=img_channels)
	if os.path.exists(pre_sf_path):
		print("Load pre-train SF-Net weights")
		sfnet.load_weights(pre_sf_path, by_name=True)

	x_sf_st = sfnet.output
	# x = Dropout(0.5)(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='sal_st_conv2d_1')(x_sf_st)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='sal_st_conv2d_2')(x)

	cb_input = Input(shape=(shape_r_out, shape_c_out, nb_gaussian))
	cb_x = Conv2D(64, (3, 3), activation='relu', padding='same', name='sal_st_cb_conv2d_1')(cb_input)
	priors = Conv2D(64, (3, 3), activation='relu', padding='same', name='sal_st_cb_conv2d_2')(cb_x)
	x = layers.concatenate([x, priors], axis=-1, name='sal_st_cb_cat')
	x_input = [sfnet.input, cb_input]

	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='sal_st_conv2d_3_cb')(x)
	x = Conv2D(1, (3, 3), activation='relu', padding='same', name='sal_st_conv2d_4')(x)

	model = Model(inputs=x_input, outputs=[x, x, x], name='salcnn_st_model')

	model.summary()
	return model

def salcnn_Dynamic_Net(time_dims=7, img_rows=480, img_cols=640, img_channels=3, pre_sf_path = ''):

	video_inputs = Input(shape=(time_dims, img_rows, img_cols, img_channels))
	# BGR to gray image, three channels
	# Subtract the mean value for dynamic stream
	x_dy_input = TimeDistributed(Conv2D(3, (1, 1), padding='same', kernel_initializer=Constant(value=(0.114,0.114,0.114, 0.587,0.587,0.587, 0.299, 0.299, 0.299)), use_bias = False), name='sal_dy_bgr2gray')(video_inputs)
	x_dy_input = TimeDistributed(Conv2D(3, (1, 1), padding='same', kernel_initializer=Constant(value=(1, 0, 0, 0, 1, 0, 0, 0, 1)), bias_initializer=Constant(value=(-103.939,-116.779,-123.68))), name='sal_dy_sub_mean')(x_dy_input)

	sfnet = salcnn_SF_Net(img_rows=img_rows, img_cols=img_cols, img_channels=img_channels)
	if os.path.exists(pre_sf_path):
		print("Load pre-train SF-Net weights")
		sfnet.load_weights(pre_sf_path, by_name=True)

	x = TimeDistributed(sfnet,name='sf_net')(x_dy_input)
	x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='sal_dy_conv3d_1')(x)
	x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='sal_dy_conv3d_2')(x)

	cb_input_3D = Input(shape=(time_dims, shape_r_out, shape_c_out, nb_gaussian))
	cb_x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='sal_dy_cb_conv2d_1'), name='sal_dy_cb_conv2d_11')(cb_input_3D)
	priors = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='sal_dy_cb_conv2d_2'), name='sal_dy_cb_conv2d_22')(cb_x)
	x = layers.concatenate([x, priors], axis=-1, name='sal_dy_cb_cat')
	x_input = [video_inputs, cb_input_3D]

	# x = layers.concatenate([x, priors], axis=-1)
	x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='sal_dy_conv3d_3_cb')(x)
	x = Conv3D(1, (3, 3, 3), activation='relu', padding='same', name='sal_dy_conv3d_4')(x)

	model = Model(inputs = x_input, outputs=[x, x, x], name='salcnn_dy_model')
	sfnet.trainable = False

	for layer in model.layers[:4]:
		layer.trainable = False

	model.summary()
	return model

#######################################################
# 3.0 TwoS-Net Fusion Networks
#######################################################

def salcnn_TwoS_Net(time_dims=7, img_rows=480, img_cols=640, img_channels=3, pre_sf_path=''):

	video_inputs = Input(shape=(time_dims, img_rows, img_cols, img_channels), name='video_input')

	# Subtract the mean value for static stream
	x_st_input = TimeDistributed(Conv2D(3, (1, 1), padding='same', kernel_initializer=Constant(value=(1, 0, 0, 0, 1, 0, 0, 0, 1)), bias_initializer=Constant(value=(-103.939,-116.779,-123.68))), name='sal_st_sub_mean')(video_inputs)

	# BGR to gray image, three channels
	# Subtract the mean value for dynamic stream
	x_dy_input = TimeDistributed(Conv2D(3, (1, 1), padding='same', kernel_initializer=Constant(value=(0.114,0.114,0.114, 0.587,0.587,0.587, 0.299, 0.299, 0.299)), use_bias = False), name='sal_dy_bgr2gray')(video_inputs)
	x_dy_input = TimeDistributed(Conv2D(3, (1, 1), padding='same', kernel_initializer=Constant(value=(1, 0, 0, 0, 1, 0, 0, 0, 1)), bias_initializer=Constant(value=(-103.939,-116.779,-123.68))), name='sal_dy_sub_mean')(x_dy_input)

	# SF-Net model
	sfnet = salcnn_SF_Net(img_rows=img_rows, img_cols=img_cols, img_channels=img_channels)
	if os.path.exists(pre_sf_path):
		print("Load pre-train SF-Net weights")
		sfnet.load_weights(pre_sf_path, by_name=True)
	x_sf_st = TimeDistributed(sfnet, name='sf_net_st')(x_st_input)
	x_sf_dy = TimeDistributed(sfnet, name='sf_net_dy')(x_dy_input)

	# St-net model
	x_st = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='sal_st_conv2d_1'),name='sal_st_conv2d_11')(x_sf_st)
	x_st = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='sal_st_conv2d_2'),name='sal_st_conv2d_22')(x_st)

	# Dy_net model
	x_dy = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='sal_dy_conv3d_1')(x_sf_dy)
	x_dy = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='sal_dy_conv3d_2')(x_dy)

	# CGP layer
	cb_inputs_st = Input(shape=(time_dims, shape_r_out, shape_c_out, nb_gaussian), name='cb_input_st')
	cb_x_st = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='sal_st_cb_conv2d_1'), name='sal_st_cb_conv2d_11')(cb_inputs_st)
	priors_st = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='sal_st_cb_conv2d_2'), 	name='sal_st_cb_conv2d_22')(cb_x_st)

	cb_inputs_dy = Input(shape=(time_dims, shape_r_out, shape_c_out, nb_gaussian), name='cb_input_dy')
	cb_x_dy = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='sal_dy_cb_conv2d_1'), name='sal_dy_cb_conv2d_11')(cb_inputs_dy)
	priors_dy = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='sal_dy_cb_conv2d_2'), name='sal_dy_cb_conv2d_22')(cb_x_dy)

	x_st = layers.concatenate([x_st, priors_st], axis=-1, name='sal_st_cb_cat')
	x_dy = layers.concatenate([x_dy, priors_dy], axis=-1, name='sal_dy_cb_cat')

	x_input = [video_inputs, cb_inputs_st, cb_inputs_dy]

	x_st = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='sal_st_conv2d_3_cb'), name='sal_st_conv2d_33_cb')(x_st)
	x_st_out = TimeDistributed(Conv2D(1, (3, 3), activation='relu', padding='same', name='sal_st_conv2d_4'), name='sal_st_conv2d_44')(x_st)

	x_dy = Conv3D(256, (3, 3, 3), activation='relu', padding='same', name='sal_dy_conv3d_3_cb')(x_dy)
	x_dy_out = Conv3D(1, (3, 3, 3), activation='relu', padding='same', name='sal_dy_conv3d_4')(x_dy)

	# Fu_net model
	x_fu = layers.concatenate([x_st_out, x_dy_out], axis=-1, name='funet_cat')

	x_fu = TimeDistributed(Conv2D(64,    (3, 3), activation='relu', padding='same', name='sal_fu_conv2d_1'), name='sal_fu_conv2d_1')(x_fu)
	x_fu = TimeDistributed(Conv2D(128,   (3, 3), activation='relu', padding='same', name='sal_fu_conv2d_2'), name='sal_fu_conv2d_2')(x_fu)
	x_fu_out = TimeDistributed(Conv2D(1, (3, 3), activation='relu', padding='same', name='sal_fu_conv2d_3'), name='sal_fu_conv2d_3')(x_fu)

	model = Model(inputs=x_input, outputs=[x_fu_out,x_fu_out,x_fu_out], name='salcnn_fu_model')
	sfnet.trainable = False

	for layer in model.layers[:4]:
		layer.trainable = False

	model.summary()
	return model

