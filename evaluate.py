import os
import argparse
import tflearn
from datetime import datetime
import numpy as np
from numpy import genfromtxt
from tflearn.data_preprocessing import ImagePreprocessing

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', action="store_true", default=False, help="path of the tflearn log directory")
args = parser.parse_args()

if __name__ == '__main__':
    dataFile = None
    if args.data:
        if not os.path.exists('data.npz'):
            raise FileNotFoundError('Please confirm path to data store to data from!')
        else:
            dataFile = np.load('data.npz')

    X = dataFile['X']
    Y = dataFile['Y']
    X_test = dataFile['X_test']
    Y_test = dataFile['Y_test']

    # Y_test = tflearn.data_utils.to_categorical(Y_test, 7)
    # X_test = X_test.reshape([-1, 48, 48, 1])

    # Residual blocks
    # 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
    n = 5

    # Real-time preprocessing of the image data
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()

    print('Building Residual Network')
    # Building Residual Network
    net = tflearn.input_data(shape=[None, 48, 48, 1], data_preprocessing=img_prep, data_augmentation=img_aug)
    net = tflearn.conv_2d(net, nb_filter=16, filter_size=3, regularizer='L2', weight_decay=0.0001)
    net = tflearn.residual_block(net, n, 16)
    net = tflearn.residual_block(net, 1, 32, downsample=True)
    net = tflearn.residual_block(net, n-1, 32)
    net = tflearn.residual_block(net, 1, 64, downsample=True)
    net = tflearn.residual_block(net, n-1, 64)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)

    # Regression
    net = tflearn.fully_connected(net, 7, activation='softmax')
    adam = tflearn.optimizers.adam()
    net = tflearn.regression(net, optimizer=adam,
                             loss='categorical_crossentropy')
    # Training
    model = tflearn.DNN(net, checkpoint_path='models/model_fer',
                        max_checkpoints=20, tensorboard_verbose=2,
                        clip_gradients=0.)

    model.load('model.tfl')

    # Predict
    print(model.predict(X_test[0]))
    print(Y_test[0])


