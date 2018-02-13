from __future__ import absolute_import
from __future__ import print_function

import copy
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from six.moves import xrange

from cleverhans.utils import other_classes
from cleverhans.utils_tf import batch_eval, model_argmax
from cleverhans.attacks_tf import (jacobian_graph, jacobian,
                                   apply_perturbations, saliency_map)
import keras.backend as K


def fgsm(x, predictions, eps, clip_min=None, clip_max=None, y=None):
    """
    Computes symbolic TF tensor for the adversarial samples. This must
    be evaluated with a session.run call.
    :param x: the input placeholder
    :param predictions: the model's output tensor
    :param eps: the epsilon (input variation parameter)
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :param y: the output placeholder. Use None (the default) to avoid the
            label leaking effect.
    :return: a tensor for the adversarial example
    """

    # Compute loss
    if y is None:
        # In this case, use model predictions as ground truth
        y = tf.to_float(
            tf.equal(predictions,
                     tf.reduce_max(predictions, 1, keep_dims=True)))
    y = y / tf.reduce_sum(y, 1, keep_dims=True)
    logits, = predictions.op.inputs
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    )

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    # Take sign of gradient
    signed_grad = tf.sign(grad)

    # Multiply by constant epsilon
    scaled_signed_grad = eps * signed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_signed_grad)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x


def jsma(sess, x, predictions, grads, sample, target, theta, gamma,
         increase, nb_classes, clip_min, clip_max, verbose=False):
    """
    TensorFlow implementation of the jacobian-based saliency map method (JSMA).
    :param sess: TF session
    :param x: the input placeholder
    :param predictions: the model's symbolic output (linear output,
        pre-softmax)
    :param sample: numpy array with sample input
    :param target: target class for sample input
    :param theta: delta for each feature adjustment
    :param gamma: a float between 0 - 1 indicating the maximum distortion
        percentage
    :param increase: boolean; true if we are increasing pixels, false otherwise
    :param nb_classes: integer indicating the number of classes in the model
    :param clip_min: optional parameter that can be used to set a minimum
                    value for components of the example returned
    :param clip_max: optional parameter that can be used to set a maximum
                    value for components of the example returned
    :param verbose: boolean; whether to print status updates or not
    :return: an adversarial sample
    """

    # Copy the source sample and define the maximum number of features
    # (i.e. the maximum number of iterations) that we may perturb
    adv_x = copy.copy(sample)
    # count the number of features. For MNIST, 1x28x28 = 784; for
    # CIFAR, 3x32x32 = 3072; etc.
    nb_features = np.product(adv_x.shape[1:])
    # reshape sample for sake of standardization
    original_shape = adv_x.shape
    adv_x = np.reshape(adv_x, (1, nb_features))
    # compute maximum number of iterations
    max_iters = np.floor(nb_features * gamma / 2)
    if verbose:
        print('Maximum number of iterations: {0}'.format(max_iters))

    # Compute our initial search domain. We optimize the initial search domain
    # by removing all features that are already at their maximum values (if
    # increasing input features---otherwise, at their minimum value).
    if increase:
        search_domain = set([i for i in xrange(nb_features)
                             if adv_x[0, i] < clip_max])
    else:
        search_domain = set([i for i in xrange(nb_features)
                             if adv_x[0, i] > clip_min])

    # Initialize the loop variables
    iteration = 0
    adv_x_original_shape = np.reshape(adv_x, original_shape)
    current = model_argmax(sess, x, predictions, adv_x_original_shape, feed={K.learning_phase(): 0})

    # Repeat this main loop until we have achieved misclassification
    while (current != target and iteration < max_iters and
           len(search_domain) > 1):
        # Reshape the adversarial example
        adv_x_original_shape = np.reshape(adv_x, original_shape)

        # Compute the Jacobian components
        grads_target, grads_others = jacobian(sess, x, grads, target,
                                              adv_x_original_shape,
                                              nb_features, nb_classes,
                                              feed={K.learning_phase(): 0})

        # Compute the saliency map for each of our target classes
        # and return the two best candidate features for perturbation
        i, j, search_domain = saliency_map(
            grads_target, grads_others, search_domain, increase)

        # Apply the perturbation to the two input features selected previously
        adv_x = apply_perturbations(
            i, j, adv_x, increase, theta, clip_min, clip_max)

        # Update our current prediction by querying the model
        current = model_argmax(sess, x, predictions, adv_x_original_shape, feed={K.learning_phase(): 0})

        # Update loop variables
        iteration += 1

        # This process may take a while, so outputting progress regularly
        if iteration % 5 == 0 and verbose:
            msg = 'Current iteration: {0} - Current Prediction: {1}'
            print(msg.format(iteration, current))

    # Compute the ratio of pixels perturbed by the algorithm
    percent_perturbed = float(iteration * 2) / nb_features

    # Report success when the adversarial example is misclassified in the
    # target class
    if current == target:
        if verbose:
            print('Successful')
        return np.reshape(adv_x, original_shape), 1, percent_perturbed
    else:
        if verbose:
            print('Unsuccesful')
        return np.reshape(adv_x, original_shape), 0, percent_perturbed


def fast_gradient_sign_method(sess, model, X, Y, eps, clip_min=None,
                              clip_max=None, batch_size=256):
    """
    TODO
    :param sess:
    :param model: predictions or after-softmax
    :param X:
    :param Y:
    :param eps:
    :param clip_min:
    :param clip_max:
    :param batch_size:
    :return:
    """
    # Define TF placeholders for the input and output
    x = tf.placeholder(tf.float32, shape=(None,) + X.shape[1:])
    y = tf.placeholder(tf.float32, shape=(None,) + Y.shape[1:])
    adv_x = fgsm(
        x, model(x), eps=eps,
        clip_min=clip_min,
        clip_max=clip_max, y=y
    )
    X_adv, = batch_eval(
        sess, [x, y], [adv_x],
        [X, Y], feed={K.learning_phase(): 0},
        args={'batch_size': batch_size}
    )

    return X_adv


def basic_iterative_method(sess, model, X, Y, eps, eps_iter, nb_iter=50,
                           clip_min=None, clip_max=None, batch_size=256):
    """
    TODO
    :param sess:
    :param model: predictions or after-softmax
    :param X:
    :param Y:
    :param eps:
    :param eps_iter:
    :param nb_iter:
    :param clip_min:
    :param clip_max:
    :param batch_size:
    :return:
    """
    # Define TF placeholders for the input and output
    x = tf.placeholder(tf.float32, shape=(None,)+X.shape[1:])
    y = tf.placeholder(tf.float32, shape=(None,)+Y.shape[1:])
    # results will hold the adversarial inputs at each iteration of BIM;
    # thus it will have shape (nb_iter, n_samples, n_rows, n_cols, n_channels)
    results = np.zeros((nb_iter, X.shape[0],) + X.shape[1:])
    # Initialize adversarial samples as the original samples, set upper and
    # lower bounds
    X_adv = X
    X_min = X_adv - eps
    X_max = X_adv + eps
    print('Running BIM iterations...')
    # "its" is a dictionary that keeps track of the iteration at which each
    # sample becomes misclassified. The default value will be (nb_iter-1), the
    # very last iteration.
    def f(val):
        return lambda: val
    its = defaultdict(f(nb_iter-1))
    # Out keeps track of which samples have already been misclassified
    out = set()
    for i in tqdm(range(nb_iter)):
        adv_x = fgsm(
            x, model(x), eps=eps_iter,
            clip_min=clip_min, clip_max=clip_max, y=y
        )
        X_adv, = batch_eval(
            sess, [x, y], [adv_x],
            [X_adv, Y], feed={K.learning_phase(): 0},
            args={'batch_size': batch_size}
        )
        X_adv = np.maximum(np.minimum(X_adv, X_max), X_min)
        results[i] = X_adv
        # check misclassifieds
        predictions = model.predict_classes(X_adv, batch_size=512, verbose=0)
        misclassifieds = np.where(predictions != Y.argmax(axis=1))[0]
        for elt in misclassifieds:
            if elt not in out:
                its[elt] = i
                out.add(elt)

    return its, results


def saliency_map_method(sess, model, X, Y, theta, gamma, clip_min=None,
                        clip_max=None):
    """
    TODO
    :param sess:
    :param model: predictions or after-softmax
    :param X:
    :param Y:
    :param theta:
    :param gamma:
    :param clip_min:
    :param clip_max:
    :return:
    """
    nb_classes = Y.shape[1]
    # Define TF placeholder for the input
    x = tf.placeholder(tf.float32, shape=(None,) + X.shape[1:])
    # Define model gradients
    grads = jacobian_graph(model(x), x, nb_classes)
    X_adv = np.zeros_like(X)
    for i in tqdm(range(len(X))):
        current_class = int(np.argmax(Y[i]))
        target_class = np.random.choice(other_classes(nb_classes, current_class))
        X_adv[i], _, _ = jsma(
            sess, x, model(x), grads, X[i:(i+1)], target_class, theta=theta,
            gamma=gamma, increase=True, nb_classes=nb_classes,
            clip_min=clip_min, clip_max=clip_max
        )

    return X_adv

