import torch
from torch import nn
from PIL import Image
from sklearn.metrics import r2_score
from torch.nn.modules.loss import _Loss, _assert_no_grad
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class ImageToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, image):

        image_array = np.array(image)
        try:
            tensor = torch.from_numpy(image_array).permute(2, 0, 1)
            tensor = tensor.float()
        except:
            tensor = torch.from_numpy(np.expand_dims(image_array, axis=2)).permute(2, 0, 1)
            tensor = tensor.float()
        # put it from HWC to CHW format
        return tensor


class TensorToImage(object):
    """Converts a Tensor (C x H x W) to a numpy.ndarray of shape (H x W x C)."""

    def __call__(self, tensor):
        image_array = tensor.numpy()
        if len(image_array.shape) == 4:
            image_array = image_array[0, :, :, :]

        image_array = np.transpose(image_array, (1, 2, 0))
        return image_array


class LogRMSELoss(_Loss):
    """
    implements the log root mean square error
    """

    def __init__(self, size_average=True, reduce=True):
        super(LogRMSELoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, predicted, target):
        _assert_no_grad(target)
        diff = (torch.log1p(predicted) - torch.log1p(target)) ** 2
        if not self.reduce:
            return diff
        loss = torch.sqrt(torch.mean(diff)) if self.size_average else torch.sqrt(torch.sum(diff))
        return loss


class RMSELoss(_Loss):
    """
    implements the root mean square error
    """

    def __init__(self, size_average=True, reduce=True):
        super(RMSELoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, predicted, target):
        _assert_no_grad(target)
        diff = (predicted - target) ** 2
        if not self.reduce:
            return diff
        loss = torch.sqrt(torch.mean(diff)) if self.size_average else torch.sqrt(torch.sum(diff))
        return loss


class ScaleInvMSELoss(_Loss):
    """
    implements the scale invariant mean square error
    """

    def __init__(self):
        super(ScaleInvMSELoss, self).__init__()

    def forward(self, predicted, target):
        _assert_no_grad(target)

        first_log = torch.log(predicted + 1e-6)
        second_log = torch.log(target + 1e-6)
        log_term = torch.mean(torch.pow(first_log - second_log, 2))
        sc_inv_term = torch.pow(torch.mean((first_log - second_log)), 2)
        loss = log_term - sc_inv_term

        return loss


class AbsRelativeLoss(_Loss):
    """
    implements the absolute relative loss
    """

    def __init__(self):
        super(AbsRelativeLoss, self).__init__()

    def forward(self, predicted, target):
        _assert_no_grad(target)
        diff = torch.div(torch.abs(target - predicted), predicted + 1)

        loss = torch.mean(diff)
        return loss


def same_scale(predicted, target, loss_function):
    """
    :param predicted: predicted output
    :param target: real target
    :param loss_function: loss function to be used
    :return: return the loss of a sample, the target is resized
             to match the size of the predicted output
    """

    _, _, h, w = predicted.size()
    th, tw = target.size()[-2:]
    target_scaled = target
    if h != th or w != tw:
        target_scaled = nn.functional.adaptive_avg_pool2d(target, (h, w))

    return loss_function(predicted, target_scaled)


def loss_metric(depth, target, loss='L1'):
    """
    :param depth: predicted output
    :param target: real target
    :param loss: string identifier of the type of loss to use
    :return: loss value using a specific loss function
    """

    if type(loss) is str:
        assert (loss in ['L1', 'MSE', 'SmoothL1', 'RMSE', 'logRMSE', 'scaleINV', 'absREL', 'cross_entropy'])

        if loss == 'L1':
            loss_function = nn.L1Loss()
        elif loss == 'MSE':
            loss_function = nn.MSELoss()
        elif loss == 'SmoothL1':
            loss_function = nn.SmoothL1Loss()
        elif loss == 'RMSE':
            loss_function = RMSELoss()
        elif loss == 'logRMSE':
            loss_function = LogRMSELoss()
        elif loss == 'scaleINV':
            loss_function = ScaleInvMSELoss()
        elif loss == 'absREL':
            loss_function = AbsRelativeLoss()
        elif loss == 'cross_entropy':
            loss_function = nn.CrossEntropyLoss()

    else:
        loss_function = loss

    loss_output = same_scale(depth, target, loss_function)
    return loss_output


def coeff_determination(predicted, target):
    """
    computes the coefficient of determination
    as a metric for a regression task
    """
    size_batch, _, height, width = predicted.size()

    # scaling the target size as the size of the predicted
    target = torch.nn.functional.adaptive_avg_pool2d(target, (height, width))

    batch_size = predicted.data.shape[0]
    predicted_np = predicted.data.view(batch_size, -1)
    target_np = target.data.view(batch_size, -1)

    r2_score_list = []
    for pred, targ in zip(predicted_np, target_np):
        # computing the coefficient of determination
        score = r2_score(pred, targ)
        r2_score_list.append(score)

    return np.mean(r2_score_list)


def plot_history(history):
    """
    Plots accuracy and loss of training and validation sets wrt epochs
    """
    n_epochs = len(history['train_loss'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax_loss = axes[0]
    ax_acc = axes[1]
    ax_loss.plot(np.arange(0, n_epochs), history['train_loss'], label="train_loss")
    ax_loss.plot(np.arange(0, n_epochs), history['val_loss'], label="val_loss")
    ax_loss.set_title("Training and Validation loss")
    ax_loss.set_xlabel("Epoch #")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_acc.plot(np.arange(0, n_epochs), history['train_acc'], label="train_acc")
    ax_acc.plot(np.arange(0, n_epochs), history['val_acc'], label="val_acc")
    ax_acc.set_title("Training and Validation accuracy")
    ax_acc.set_xlabel("Epoch #")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    plt.tight_layout()
    plt.show()


