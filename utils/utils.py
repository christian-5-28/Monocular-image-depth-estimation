import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.callbacks import Callback

from datetime import datetime

import os


def coeff_determination(y_true, y_pred):
    """
    metric useful for testing the wellness of a
    model on a regression task
    """

    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def log_rmse_loss(y_true, y_pred):
    """
    logrmse loss for keras compliant implementation
    """
    # y_true = tf.Print(y_true, [y_true], message='y_true', summarize=30)
    # y_pred = tf.Print(y_pred, [y_pred], message='y_pred', summarize=30)
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)

    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1) + 0.00001)


def rmse_metric(y_true, y_pred):
    """
    root mean square error implementation,
    used as metric for test the wellness of
    a model on a regression task
    """
    y_true = y_true
    y_pred = y_pred
    rmse = K.sqrt(K.mean(K.square((y_true - y_pred))))
    return rmse


def logrmse_metric(y_true, y_pred):
    """
    log root mean square error implementation,
    used as metric for test the wellness of
    a model on a regression task
    """

    first_log = K.log(y_pred + 1.)
    second_log = K.log(y_true + 1.)

    logrmse = K.sqrt(K.mean(K.square((first_log - second_log))))
    return logrmse


#### NUMPY COMPLIANT METRICS ####

def rmse_metric_np(y_true, y_pred):
    """
    numpy compliant rmse metric
    """

    y_true = y_true
    y_pred = y_pred
    diff = y_true - y_pred
    square = np.square(diff)
    mean = np.mean(np.mean(square, axis=0))
    rmse_error = np.sqrt(mean + 0.00001)
    return rmse_error


def logrmse_metric_np(y_true, y_pred):
    """
    numpy compliant logrmse metric
    """

    y_true = y_true
    y_pred = y_pred
    first_log = np.log(y_pred + 1.)
    second_log = np.log(y_true + 1.)
    return np.sqrt(np.mean(np.mean(np.square(first_log - second_log), axis=0)))


class BatchCallback(Callback):
    """
    callback to print logs at
    the end of a batch
    """

    def on_batch_end(self, batch, logs={}):
        print(logs)


def prepare_dirs(config):
    """
    creates the directories to store
    the logs, the model weights from the checkpoint
    and the tensorboard logs
    """

    # creating the name for the model directory
    config.model_name = "{}_{}_{}_{}_test_dirs_{}_{}".format(config.exp_name,
                                                             config.data_main_dir,
                                                             config.input_height,
                                                             config.input_width,
                                                             config.data_test_dirs,
                                                             datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    config.model_dir = os.path.join(config.log_dir, config.model_name)

    # creating the name for the tensorboard directory
    config.tensorboard_dir = os.path.join(config.log_dir, config.model_name, 'tensorboard')

    # for each of the three names, create a directory if it does not already exist
    for path in [config.log_dir, config.model_dir, config.tensorboard_dir]:
        if not os.path.exists(path):
            if config.is_train:
                os.makedirs(path)
