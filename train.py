# A large portion of this code comes from the tf-learn example page:
# https://github.com/tflearn/tflearn/blob/master/examples/images/residual_network_cifar10.py
# https://github.com/safreita1/Resnet-Emotion-Recognition/blob/master/emotion_recognition.py

import os
import argparse
import tflearn
from datetime import datetime
import numpy as np
from numpy import genfromtxt
from tflearn.data_preprocessing import ImagePreprocessing

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', required=True, help="path of the csv file")
parser.add_argument('-l', '--log', required=True, help="path of the tflearn log directory")
parser.add_argument('-L', '--load', action="store_true", help="path of the tflearn log directory")
parser.add_argument('-s', '--save', action="store_true", help="save training and test data")

args = parser.parse_args()

if __name__ == "__main__":
    print('Starting Emotion Recognition Model Training at {0}'.format(str(datetime.now())))

    if not os.path.exists(args.log):
        os.makedirs(args.log)
    if not os.path.exists(args.file):
        raise FileNotFoundError('Please confirm path to training data file!')

    savefile = None
    loadfile = None
    if not args.save and args.load:
        if not os.path.exists('data.npz'):
            raise FileNotFoundError('Please confirm path to data store to load from!')
        else:
            loadfile = np.load('data.npz')
    elif args.save and not args.load:
        savefile = 'data.npz'

    # Residual blocks
    # 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
    n = 5

    # Data loading and pre-processing
    if args.save:
        data = np.asarray(np.genfromtxt(args.file, delimiter=',',  dtype=None))
        label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        labels = data[1:,0].astype(np.int32)
        image_buffer = data[1:,1]
        images = np.array([np.fromstring(image, np.uint8, sep=' ') for image in image_buffer])
        usage = data[1:,2]
        dataset = zip(labels, images, usage)

        X = np.asarray([])
        Y = np.asarray([])
        X_test = np.asarray([])
        Y_test = np.asarray([])

        print('Pre-Processing images and labels at {0}'.format(str(datetime.now())))
        for i, data in enumerate(dataset):
          if data[-1] == b'Training':
            X = np.append(X, data[1])
            Y = np.append(Y, data[0])
          elif data[-1] in [b'PublicTest', b'PrivateTest']:
            X_test = np.append(X_test, data[1])
            Y_test = np.append(Y_test, data[0])

        print('Saving training and test data to {0} at {1}'.format(savefile, str(datetime.now())))
        np.savez(savefile, X=X, Y=Y, X_test=X_test, Y_test=Y_test)
    elif args.load:
        print('Loading training and test data from {0} at {1}'.format(loadfile, str(datetime.now())))
        X = savefile['X']
        Y = savefile['Y']
        X_test = savefile['X_test']
        Y_test = savefile['Y_test']


    # Reshape the images into 48x48
    X = X.reshape([-1, 48, 48, 1])
    X_test = X_test.reshape([-1, 48, 48, 1])

    # One hot encode the labels
    Y = tflearn.data_utils.to_categorical(Y, 7)
    Y_test = tflearn.data_utils.to_categorical(Y_test, 7)

    # Real-time preprocessing of the image data
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Real-time data augmentation
    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_flip_leftright()

    print('Building Residual Network at {0}'.format(str(datetime.now())))
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
    adam = tflearn.optimizers.Adam()
    net = tflearn.regression(net, optimizer=adam,
                             loss='categorical_crossentropy')
    # Training
    model = tflearn.DNN(net, checkpoint_path='models/model_fer',
                        max_checkpoints=20, tensorboard_verbose=2,
                        clip_gradients=0., tensorboard_dir=args.log)

    print('Commencing model fitting at {0}'.format(str(datetime.now())))
    model.fit(X, Y, n_epoch=150, snapshot_epoch=False, snapshot_step=500,
              show_metric=True, batch_size=128, shuffle=True, run_id='train_fer')

    score = model.evaluate(X_test, Y_test)
    print('Test accuracy: {0}'.format(score))

    model.save('model.tfl')

    print('Finished training at {0}'.format(str(datetime.now())))
    