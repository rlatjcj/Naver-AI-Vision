from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import keras.backend as K
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Input
from keras.layers import SeparableConv2D
from keras.layers import Add
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Conv2DTranspose
from keras.layers import Concatenate
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Lambda
from keras.models import Model


def VGG(input_shape=None, classes=1000, mode='classification', **kwargs):
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    r_x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(r_x)
    s_x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(s_x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    if mode == 'siamese':
        model = Model(img_input, s_x, name='vgg16')
    elif mode == 'classification':
        model = Model(img_input, x, name='vgg16')
    elif mode == 'rmac':
        model = Model(img_input, r_x, name='vgg16')

    return model


def Xception(input_shape=None, classes=1000, mode='classification', **kwargs):
    img_input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = Add()([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = Add()([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = Add()([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)

        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)

        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = Add()([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)

    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = Add()([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    s_x = Activation('relu', name='block14_sepconv2_act')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(s_x)
    x = Dense(classes, name='predictions')(x)
    x = Activation('softmax')(x)

    if mode == 'siamese':
        s_x = Flatten()(s_x)
        model = Model(img_input, s_x, name='xception')
    elif mode == 'classification':
        model = Model(img_input, x, name='xception')
    
    return model

def Siamese(input_shape=None, model=None):
    query_input = Input(shape=input_shape)
    refer_input = Input(shape=input_shape)

    query_output = model(query_input)
    refer_output = model(refer_input)

    both = Lambda(lambda x: K.abs(x[0]-x[1]))([query_output, refer_output])
    prediction = Dense(2, activation='softmax')(both)
    siamese_net = Model([query_input, refer_input], prediction, name='siamese')

    return siamese_net


def RMAC(input_shape=None, model=None, num_rois=None):
    from RoiPooling import RoiPooling
    from keras.layers import TimeDistributed

    in_roi = Input(shape=(num_rois, 4), name='input_roi')
    x = RoiPooling([1], num_rois)([model.layers[-1].output, in_roi])    # ROI pooling
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='norm1')(x)    # Normalization
    
    x = TimeDistributed(Dense(512, name='pca',
                        kernel_initializer='identity',
                        bias_initializer='zeros'))(x)                   # PCA
    x = Lambda(lambda x: K.l2_normalize(x, axis=2), name='pca_norm')(x) # Normalization

    rmac = Lambda(lambda x: K.sum(x, axis=1), output_shape=(512,), name='rmac')(x)  # Addition
    rmac_norm = Lambda(lambda x: K.l2_normalize(x, axis=1), name='rmac_norm')(rmac) # Normalization

    rmac_model = Model([model.input, in_roi], rmac_norm)

    return rmac_model


def Triple_Siamese(input_shape=None, rmac=None, num_rois=None):
    def triplet_loss(q, r, ir, m=0.1):
        triplet = m + K.sum(K.square(q-r)) + K.sum(K.square(q-ir))
        loss = K.maximum(0., triplet) / 2
        return loss

    query = Input(shape=input_shape, name='query')
    relevant = Input(shape=input_shape, name='relevant')
    irrelevant = Input(shape=input_shape, name='irrelevant')
    in_roi = Input(shape=(num_rois, 4), name='input_roi')

    query_feature = rmac([query, in_roi])
    relevant_feature = rmac([relevant, in_roi])
    irrelevant_feature = rmac([irrelevant, in_roi])

    # loss = Lambda(lambda x: triplet_loss(*x))([query_feature, relevant_feature, irrelevant_feature])
    model = Model([query, relevant, irrelevant, in_roi], [query_feature, relevant_feature, irrelevant_feature])

    return model



if __name__ == "__main__":
    model = Xception((256, 256, 3))
    model.summary()
    # from get_regions import rmac_regions, get_size_vgg_feat_map
    # from keras.utils.vis_utils import plot_model
    # vgg = VGG((256, 256, 3), mode='rmac')
    # Wmap, Hmap = get_size_vgg_feat_map(256, 256)
    # regions = rmac_regions(Wmap, Hmap, 3)
    # rmac = RMAC((256, 256, 3), vgg, len(regions))
    # model = Triple_Siamese((256, 256, 3), rmac, len(regions))
    # model.summary()
    # print(model.layers[-2])
    # plot_model(model, './triple_siamese.png', True, True)