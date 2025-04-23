from torch import nn
import numpy
import torch
from scipy.ndimage import (
    _ni_support,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure
)


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
            pred.double().sum() + truth.double().sum() + 1
        )

        return bce_loss + (1 - dice_coef)
    
class BCEDiceLossWithLogits(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
      pred = torch.sigmoid(input).reshape(-1)  # convert logits to probabilities
      truth = target.reshape(-1)

      # BCE loss with logits (more stable)
      bce_loss = nn.BCEWithLogitsLoss()(input.reshape(-1), truth).double()

      # Dice coefficient on probabilities
      dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
          pred.double().sum() + truth.double().sum() + 1
      )

      return bce_loss + ( 1 - dice_coef)
    

# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class MetricTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
def jaccard_index(input, target):

    intersection = (input * target).long().sum().data.cpu()[0]
    union = (
        input.long().sum().data.cpu()[0]
        + target.long().sum().data.cpu()[0]
        - intersection
    )

    if union == 0:
        return float("nan")
    else:
        return float(intersection) / float(max(union, 1))


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(input, target):
    num_in_target = input.size(0)

    smooth = 1.0

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2.0 * intersection + smooth) / (pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().item()

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = numpy.atleast_1d(result.astype(numpy.bool_))
    reference = numpy.atleast_1d(reference.astype(numpy.bool_))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = numpy.asarray(voxelspacing, dtype=numpy.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == numpy.count_nonzero(result):
        raise RuntimeError(
            "The first supplied array does not contain any binary object."
        )
    if 0 == numpy.count_nonzero(reference):
        raise RuntimeError(
            "The second supplied array does not contain any binary object."
        )

    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(
        reference, structure=footprint, iterations=1
    )

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def hd95(result, reference, voxelspacing=None, connectivity=1):
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = numpy.percentile(numpy.hstack((hd1, hd2)), 95)
    return hd95

import numpy as np

def hd95_batch(results, references, voxelspacing=None, connectivity=1):

    if isinstance(results, torch.Tensor):
        results = results.cpu().numpy()
    if isinstance(references, torch.Tensor):
        references = references.cpu().numpy()

    batch_size = results.shape[0]
    hd95_list = []

    for i in range(batch_size):
        pred_i = results[i]
        ref_i = references[i]

        # check to see if the prediction contains any foreground object i.e. mask
        if np.any(pred_i) and np.any(ref_i):
            hd1 = __surface_distances(pred_i, ref_i, voxelspacing, connectivity)
            hd2 = __surface_distances(ref_i, pred_i, voxelspacing, connectivity)

            hd95_value = np.percentile(np.hstack((hd1, hd2)), 95)
        else:
            hd95_value = np.nan

        hd95_list.append(hd95_value)

    hd95_array = np.array(hd95_list)
    hd95_array = hd95_array[~np.isnan(hd95_array)]  # remove NaNs

    if len(hd95_array) == 0:
        return float('nan') 

    return float(hd95_array.mean())

