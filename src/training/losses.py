import torch
import torch.nn as nn


def dice_loss(y_pred, y_true, use_sigmoid=True):
    """Dice loss: harmonic mean of precision and recall (FPs and FNs are weighted equally). Only for 1 output channel.

    :param y_pred: Prediction [batch size, channels, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, channels, height, width].
        :type y_true:
    :param use_sigmoid: Apply sigmoid activation function to the prediction y_pred.
        :type use_sigmoid: bool
    :return:
    """

    # Avoid division by zero
    smooth = 1.

    # Flatten ground truth
    gt = y_true.contiguous().view(-1)

    if use_sigmoid:  # Apply sigmoid activation to prediction and flatten prediction
        pred = torch.sigmoid(y_pred)
        pred = pred.contiguous().view(-1)
    else:
        pred = y_pred.contiguous().view(-1)

    # Calculate Dice loss
    pred_gt = torch.sum(gt * pred)
    loss = 1 - (2. * pred_gt + smooth) / (torch.sum(gt ** 2) + torch.sum(pred ** 2) + smooth)

    return loss


def bce_dice(y_pred, y_true):
    """ Sum of binary cross-entropy loss and Dice loss. Only for 1 output channel.

    :param y_pred: Prediction [batch size, channels, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, height, width].
        :type y_true:
    :return:
    """
    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(y_pred, y_true) + 0.5 * dice_loss(y_pred, y_true)

    return loss
