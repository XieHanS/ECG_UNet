# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:49:32 2019

@author: Winham

Unet.py: Unet模型定义


"""

from keras.models import Model
from keras.layers import Input, core, Dropout, concatenate
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D



def Unet(nClasses, input_length,optimizer=None,  nChannels=1):
    inputs = Input((input_length, nChannels))
    # (None,1800,1)--->(None,1800,16)
    conv1 = Conv1D(16, 32, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # (None,1800,16)--->(None,1800,16)
    conv1 = Conv1D(16, 32, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # (None,1800,16)--->(None,900,16)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    # (None,900,16)--->(None,900,32)
    conv2 = Conv1D(32, 32, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    # (None,900,32)--->(None,900,32)
    conv2 = Dropout(0.2)(conv2)
    # (None,900,32)--->(None,900,32)
    conv2 = Conv1D(32, 32, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    # (None,900,32)--->(None,450,32)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    #(None, 450, 32)--->(None,450,64)
    conv3 = Conv1D(64, 32, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    # (None, 450, 64)--->(None,450,64)
    conv3 = Conv1D(64, 32, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # (None, 450, 64)--->(None,225,64)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    #(None,225,64)--->(None,225,128)
    conv4 = Conv1D(128, 32, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    # (None,225,128)--->(None,225,128)
    conv4 = Dropout(0.5)(conv4)
    # (None,225,128)--->(None,225,128)
    conv4 = Conv1D(128, 32, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    #(None,225,128)--->(None,450,64)
    up1 = Conv1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv4))
    #(None, 450, 64)--->(None,450,128)
    merge1 = concatenate([up1, conv3], axis=-1)
    # (None, 450, 64)--->(None,450,64)
    conv5 = Conv1D(64, 32, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
    # (None, 450, 64)--->(None,450,64)
    conv5 = Conv1D(64, 32, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    # (None, 450, 64)--->(None,900,32)
    up2 = Conv1D(32, 2, activation='relu', padding='same', kernel_initializer = 'he_normal')(UpSampling1D(size=2)(conv5))
    # (None, 900, 32)--->(None,900,64)
    merge2 = concatenate([up2, conv2], axis=-1)
    # (None, 900, 64)--->(None,900,32)
    conv6 = Conv1D(32, 32, activation='relu', padding='same', kernel_initializer = 'he_normal')(merge2)
    conv6 = Dropout(0.2)(conv6)
    # (None, 900, 32)--->(None,900,32)
    conv6 = Conv1D(32, 32, activation='relu', padding='same')(conv6)

    # (None, 900, 32)--->(None,1800,16)
    up3 = Conv1D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv6))
    #(None,1800,16)--->(None,1800,32)
    merge3 = concatenate([up3, conv1], axis=-1)
    #(None, 1800, 32)--->(None, 1800, 16)
    conv7 = Conv1D(16, 32, activation='relu', padding='same', kernel_initializer='he_normal')(merge3)
    #(None, 1800, 16)--->(None, 1800, 16)
    conv7 = Conv1D(16, 32, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    # (None, 1800, 16)--->(None, 1800, classes)
    conv8 = Conv1D(nClasses, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    #(None, 1800, classes)--->(None,3,1800)
    conv8 = core.Reshape((nClasses, input_length))(conv8)
    #(None, 3, 1800)--->(None, 1800, classes)
    conv8 = core.Permute((2, 1))(conv8)

    conv9 = core.Activation('softmax')(conv8)

    model = Model(inputs=inputs, outputs=conv9)
    if not optimizer is None:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model


if __name__ == '__main__':
    print('\nSummarize the model:\n')
    InputLen = 2560
    model = Unet(3,InputLen)
    model.summary()

    from keras.utils import plot_model
    plot_model(model, to_file='./U_Net_1D.png')
    print('\nEnd for summary.\n')
