from __future__ import absolute_import
from __future__ import print_function

import argparse
from util import get_data, get_model, cross_entropy
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def train(dataset='mnist', batch_size=128, epochs=50):
    """
    Train one model with data augmentation: random padding+cropping and horizontal flip
    :param args: 
    :return: 
    """
    print('Data set: %s' % dataset)
    X_train, Y_train, X_test, Y_test = get_data(dataset)
    model = get_model(dataset)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy']
    )
    # training without data augmentation
    model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        validation_data=(X_test, Y_test)
    )

    # training with data augmentation
    # data augmentation
    # datagen = ImageDataGenerator(
    #     # rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True)
    # # fit parameters from data
    # datagen.fit(X_train)
    #
    # model.fit_generator(
    #     datagen.flow(X_train, Y_train, batch_size=batch_size),
    #     steps_per_epoch=len(X_train) / batch_size,
    #     epochs=epochs,
    #     verbose=1,
    #     validation_data=(X_test, Y_test))

    model.save('../data/model_%s.h5' % dataset)

def main(args):
    """
    Train model with data augmentation: random padding+cropping and horizontal flip
    :param args: 
    :return: 
    """
    assert args.dataset in ['mnist', 'cifar', 'svhn', 'all'], \
        "dataset parameter must be either 'mnist', 'cifar', 'svhn' or all"
    if args.dataset == 'all':
        for dataset in ['mnist', 'cifar', 'svhn']:
            train(dataset, args.batch_size, args.epochs)
    else:
        train(args.dataset, args.batch_size, args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar', 'svhn' or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.set_defaults(epochs=20)
    parser.set_defaults(batch_size=100)
    args = parser.parse_args()
    main(args)
