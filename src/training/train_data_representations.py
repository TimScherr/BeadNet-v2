import numpy as np
import numpy.typing as npt
import random
from scipy.ndimage import binary_dilation
from scipy.ndimage.morphology import generate_binary_structure
from skimage import measure
from typing import Union


def get_label(mask: npt.NDArray[Union[np.ubyte, np.ushort, np.uintc, np.uint]],
              label_type: str) -> npt.NDArray[np.ushort]:
    """ Calculate label image (1x1 seeds, dilated seeds, ...)

    :param mask: Mask image (contains particle markers).
    :param label_type: Label type (1x1, 1x1_dilation, 2x2, 2x2_dilation, 3x3)
    :return: Label image
    """

    if label_type == '1x1':
        label_img = mask.astype(np.uint8)
    elif label_type == '1x1_dilation':
        label_img = binary_dilation(mask, generate_binary_structure(2, 1)) > 0
        label_img = label_img.astype(np.uint8)
    elif label_type == '2x2':
        label_img = np.zeros_like(mask)
        mask = measure.label(mask, background=0)
        props = measure.regionprops(mask)
        for i in range(len(props)):
            centroid = tuple(np.round(props[i].centroid).astype(np.uint16))
            # randomly assign 3 next pixels
            h = random.randint(0, 3)
            # Check if centroid is exactly at borders (some cases will not work)
            if centroid[0] <= 0 or centroid[0] >= mask.shape[0] - 1:
                continue
            if centroid[1] <= 0 or centroid[1] >= mask.shape[1] - 1:
                continue
            if h == 0:  # go up and left
                label_img[centroid[0]-1:centroid[0]+1, centroid[1]-1:centroid[1]+1] = 1
            elif h == 1:  # go down and left
                label_img[centroid[0]:centroid[0]+2, centroid[1]-1:centroid[1]+1] = 1
            elif h == 2:  # go down and right
                label_img[centroid[0]:centroid[0]+2, centroid[1]:centroid[1]+2] = 1
            elif h == 3:  # go up and right
                label_img[centroid[0]-1:centroid[0]+1, centroid[1]:centroid[1]+2] = 1
        label_img = label_img.astype(np.uint8)
    elif label_type == '2x2_dilation':
        label_img = np.zeros_like(mask)
        mask = measure.label(mask, background=0)
        props = measure.regionprops(mask)
        for i in range(len(props)):
            centroid = tuple(np.round(props[i].centroid).astype(np.uint16))
            # randomly assign 3 next pixels
            h = random.randint(0, 3)
            # Check if centroid is exactly at borders (some cases will not work)
            if centroid[0] <= 0 or centroid[0] >= mask.shape[0] - 1:
                continue
            if centroid[1] <= 0 or centroid[1] >= mask.shape[1] - 1:
                continue
            if h == 0:  # go up and left
                label_img[centroid[0] - 1:centroid[0] + 1, centroid[1] - 1:centroid[1] + 1] = 1
            elif h == 1:  # go down and left
                label_img[centroid[0]:centroid[0] + 2, centroid[1] - 1:centroid[1] + 1] = 1
            elif h == 2:  # go down and right
                label_img[centroid[0]:centroid[0] + 2, centroid[1]:centroid[1] + 2] = 1
            elif h == 3:  # go up and right
                label_img[centroid[0] - 1:centroid[0] + 1, centroid[1]:centroid[1] + 2] = 1
        label_img = binary_dilation(label_img, generate_binary_structure(2, 1)) > 0
        label_img = label_img.astype(np.uint8)
    elif label_type == '3x3':
        label_img = binary_dilation(mask, generate_binary_structure(2, 2)) > 0
        label_img = label_img.astype(np.uint8)
    else:
        raise Exception('Label type not known')

    return label_img
