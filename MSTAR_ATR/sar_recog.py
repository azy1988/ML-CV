# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 21:39:53 2018

@author: Zhenyu An
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,Input
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, ResNet50
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.models import load_model


K.set_image_dim_ordering('th')

WEIGHTS_PATH = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
img_width, img_height = 200,200

#model = VGG16(include_top=False, weights='imagenet')

input_tensor = Input(shape=(3,img_width, img_height)) # 当使用不包括top的VGG16时，要指定输入的shape，否则会报错
model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)
print('Model loaded.')
model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')


x = model.output
x = Flatten()(x)
x = Dense(256,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(10, activation = 'softmax')(x)

model2 = Model(inputs=model.input, outputs=x)


#model2 = load_model('mstar.h5')


for layer in model2.layers[:45]: # set the first 11 layers(fine tune conv4 and conv5 block can also further improve accuracy
    layer.trainable = False
model2.compile(loss='binary_crossentropy',
              optimizer = SGD(lr=1e-3,momentum=0.9),#SGD(lr=1e-3,momentum=0.9)
              metrics=['accuracy'])





train_data_dir = 'E:\\study\\雷达数据\\mydata\\train'
validation_data_dir = 'E:\\study\\雷达数据\\mydata\\test'
#img_width, img_height = 128, 128
nb_train_samples = 2536
nb_validation_samples = 2636
epochs = 200
batch_size = 16


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        rotation_range=10.,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# 图片generator
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

#model2.load_weights('mstar.h5')

model2.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)


