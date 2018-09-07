import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import (Convolution2D, GlobalAveragePooling2D, Input, Lambda, InputLayer,
                          MaxPooling2D, merge, Merge, Concatenate, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.regularizers import l2
from IPython import embed

image_input = Input(shape=(416,416,3),name='input_1')
weights_file = open('yolo2.weights','rb')
count = 16
weights_file.read(16)
"""
convolution_0
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(32,),dtype='float32',buffer=weights_file.read(32*4))
bn_list = np.ndarray(shape=(3, 32),dtype='float32',buffer=weights_file.read(32*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(32,3,3,3),dtype='float32',buffer=weights_file.read(32*3*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(32,3,3,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_1')(image_input)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_1')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_1')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (32 + 32*3 + 3*3*3*32)*4
"""
maxpool_0
size=2
stride=2
"""
tmp = MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='same',name='maxpooling2d_1')(tmp)
"""
convolutional_1
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(64,),dtype='float32',buffer=weights_file.read(64*4))
bn_list = np.ndarray(shape=(3, 64),dtype='float32',buffer=weights_file.read(64*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(64,32,3,3),dtype='float32',buffer=weights_file.read(64*32*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(64,3,3,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_2')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_2')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_2')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (64 + 64*3 + 3*3*64*32)*4
"""
maxpool_1
size=2
stride=2
"""
tmp = MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='same',name='maxpooling2d_2')(tmp)
"""
convolutional_2
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(128,),dtype='float32',buffer=weights_file.read(128*4))
bn_list = np.ndarray(shape=(3, 128),dtype='float32',buffer=weights_file.read(128*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(128,64,3,3),dtype='float32',buffer=weights_file.read(128*64*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(128,3,3,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_3')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_3')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_3')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (128 + 128*3 + 3*3*128*64)*4
"""
convolutional_3
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(64,),dtype='float32',buffer=weights_file.read(64*4))
bn_list = np.ndarray(shape=(3, 64),dtype='float32',buffer=weights_file.read(64*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(64,128,1,1),dtype='float32',buffer=weights_file.read(64*128*1*1*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(64,1,1,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_4')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_4')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_4')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (64 + 64*3 + 1*1*64*128)*4
"""
convolutional_4
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(128,),dtype='float32',buffer=weights_file.read(128*4))
bn_list = np.ndarray(shape=(3, 128),dtype='float32',buffer=weights_file.read(128*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(128,64,3,3),dtype='float32',buffer=weights_file.read(128*64*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(128,3,3,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_5')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_5')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_5')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (128 + 128*3 + 3*3*128*64)*4
"""
maxpool_2
size=2
stride=2
"""
tmp = MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='same',name='maxpooling2d_3')(tmp)
"""
convolutional_5
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(256,),dtype='float32',buffer=weights_file.read(256*4))
bn_list = np.ndarray(shape=(3, 256),dtype='float32',buffer=weights_file.read(256*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(256,128,3,3),dtype='float32',buffer=weights_file.read(256*128*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(256,3,3,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_6')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_6')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_6')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (256 + 256*3 + 3*3*256*128)*4
"""
convolutional_6
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(128,),dtype='float32',buffer=weights_file.read(128*4))
bn_list = np.ndarray(shape=(3, 128),dtype='float32',buffer=weights_file.read(128*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(128,256,1,1),dtype='float32',buffer=weights_file.read(128*256*1*1*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(128,1,1,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_7')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_7')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_7')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (128 + 128*3 + 1*1*128*256)*4
"""
convolutional_7
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(256,),dtype='float32',buffer=weights_file.read(256*4))
bn_list = np.ndarray(shape=(3, 256),dtype='float32',buffer=weights_file.read(256*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(256,128,3,3),dtype='float32',buffer=weights_file.read(256*128*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(256,3,3,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_8')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_8')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_8')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (256 + 256*3 + 3*3*256*128)*4
"""
maxpool_3
size=2
stride=2
"""
tmp = MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='same',name='maxpooling2d_4')(tmp)
"""
convolutional_8
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(512,),dtype='float32',buffer=weights_file.read(512*4))
bn_list = np.ndarray(shape=(3, 512),dtype='float32',buffer=weights_file.read(512*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(512,256,3,3),dtype='float32',buffer=weights_file.read(512*256*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(512,3,3,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_9')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_9')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_9')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (512 + 512*3 + 3*3*512*256)*4
"""
convolutional_9
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(256,),dtype='float32',buffer=weights_file.read(256*4))
bn_list = np.ndarray(shape=(3, 256),dtype='float32',buffer=weights_file.read(256*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(256,512,1,1),dtype='float32',buffer=weights_file.read(256*512*1*1*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(256,1,1,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_10')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_10')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_10')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (256 + 256*3 + 1*1*256*512)*4
"""
convolutional_10
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(512,),dtype='float32',buffer=weights_file.read(512*4))
bn_list = np.ndarray(shape=(3, 512),dtype='float32',buffer=weights_file.read(512*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(512,256,3,3),dtype='float32',buffer=weights_file.read(512*256*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(512,3,3,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_11')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_11')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_11')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (512 + 512*3 + 3*3*512*256)*4
"""
convolutional_11
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(256,),dtype='float32',buffer=weights_file.read(256*4))
bn_list = np.ndarray(shape=(3, 256),dtype='float32',buffer=weights_file.read(256*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(256,512,1,1),dtype='float32',buffer=weights_file.read(256*512*1*1*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(256,1,1,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_12')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_12')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_12')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (256 + 256*3 + 1*1*256*512)*4
"""
convolutional_12
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(512,),dtype='float32',buffer=weights_file.read(512*4))
bn_list = np.ndarray(shape=(3, 512),dtype='float32',buffer=weights_file.read(512*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(512,256,3,3),dtype='float32',buffer=weights_file.read(512*256*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(512,3,3,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_13')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_13')(tmp)
                     
#activation
image_tmp_output = LeakyReLU(alpha=0.1,name='leakyrelu_13')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (512 + 512*3 + 3*3*512*256)*4
"""
maxpool_4
size=2
stride=2
"""
tmp = MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='same',name='maxpooling2d_5')(image_tmp_output)
"""
convolutional_13
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(1024,512,3,3),dtype='float32',buffer=weights_file.read(1024*512*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(1024,3,3,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_14')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_14')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_14')(tmp)
#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (1024 + 1024*3 + 3*3*1024*512)*4
"""
convolutional_14
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(512,),dtype='float32',buffer=weights_file.read(512*4))
bn_list = np.ndarray(shape=(3, 512),dtype='float32',buffer=weights_file.read(512*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(512,1024,1,1),dtype='float32',buffer=weights_file.read(512*1024*1*1*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(512,1,1,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_15')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_15')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_15')(tmp)

count += (512 + 512*3 + 1*1*512*1024)*4
"""
convolutional_15
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(1024,512,3,3),dtype='float32',buffer=weights_file.read(1024*512*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(1024,3,3,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_16')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_16')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_16')(tmp)

count += (1024 + 1024*3 + 3*3*1024*512)*4
"""
convolutional_16
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(512,),dtype='float32',buffer=weights_file.read(512*4))
bn_list = np.ndarray(shape=(3, 512),dtype='float32',buffer=weights_file.read(512*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(512,1024,1,1),dtype='float32',buffer=weights_file.read(512*1024*1*1*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(512,1,1,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_17')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_17')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_17')(tmp)

count += (512 + 512*3 + 1*1*512*1024)*4
"""
convolutional_17
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(1024,512,3,3),dtype='float32',buffer=weights_file.read(1024*512*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(1024,3,3,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_18')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_18')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_18')(tmp)

count += (1024 + 1024*3 + 3*3*1024*512)*4
"""
convolutional_18
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(1024,1024,3,3),dtype='float32',buffer=weights_file.read(1024*1024*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(1024,3,3,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_19')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_19')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_19')(tmp)

count += (1024 + 1024*3 + 3*3*1024*1024)*4
"""
convolutional_19
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(1024,1024,3,3),dtype='float32',buffer=weights_file.read(1024*1024*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(1024,3,3,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_20')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_20')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_20')(tmp)

count += (1024 + 1024*3 + 3*3*1024*1024)*4

"""
convolutional_20
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky
"""

#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(64,),dtype='float32',buffer=weights_file.read(64*4))
bn_list = np.ndarray(shape=(3, 64),dtype='float32',buffer=weights_file.read(64*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(64,512,1,1),dtype='float32',buffer=weights_file.read(64*512*1*1*4))
weights = np.transpose(weights,(2,3,1,0))


#read for convolution
tmp2 = Convolution2D(64,1,1,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_21')(image_tmp_output)

#batchnormalization
tmp2 = BatchNormalization(weights=bn_weights,name='batch_normalization_21')(tmp2)
                     
#activation
tmp2 = LeakyReLU(alpha=0.1,name='leakyrelu_21')(tmp2)
count += (64 + 64*3 + 1*1*64*512)*4

"""
def fun(x):
    import tensorflow as tf
    block_size = 2
    _, height, width, depth = x.get_shape()
    reduced_height = height // block_size
    reduced_width = width // block_size

    result = tf.reshape(x, (-1, reduced_height, block_size, reduced_width, block_size, depth))
    result = tf.transpose(result, [0, 1, 3, 2, 4, 5])
    result = tf.reshape(result, (-1, reduced_height, reduced_width, block_size * block_size * depth))
    return result
"""


#tmp2 = Lambda(fun,output_shape=(13,13,256), name='space2depth')(tmp2)
tmp2 = Reshape((13,13,256),name='leakyrelu21_reshape')(tmp2)
tmp = Concatenate(axis=-1)([tmp2, tmp])



#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(1024,1280,3,3),dtype='float32',buffer=weights_file.read(1024*1280*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(1024,3,3,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights], 
              bias=False,
              W_regularizer=l2(0.0005),
              name='conv2d_22')(tmp)
#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_22')(tmp)
                     
#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_22')(tmp)

count += (1024 + 1024*3 + 3*3*1024*1280)*4

#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(425,),dtype='float32',buffer=weights_file.read(425*4))
weights = np.ndarray(shape=(425,1024,1,1),dtype='float32',buffer=weights_file.read(425*1024*1*1*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Convolution2D(425,1,1,
              subsample=(1,1),
              border_mode='same',
              activation=None, 
              weights=[weights,bias], 
              bias=True,
              W_regularizer=l2(0.0005),
              name='conv2d_23')(tmp)

count += (1*1*1024*425)*4

model = Model(inputs=image_input, outputs=tmp)
model.save('yolo.h5')
weights_file.close()
