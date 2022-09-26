import numpy as np
import numpy.typing as npt
from skimage import measure
from typing import Any, Tuple, List


def centroid_postprocessing(prediction: npt.NDArray[float],
                            intensity_image: npt.NDArray[Any] = None) -> Tuple[npt.NDArray[bool], int, List[int]]:
    """ Post-processing for centroid extraction.

    :param prediction: Marker prediction.
    :param intensity_image: Intensity image for weighted centroid calculation.
    :return: Centroid image/array, particle counts.
    """
    # Binarize the channels
    prediction_bin = prediction > 0.5

    # Label markers
    markers = measure.label(prediction_bin, connectivity=1, background=0)

    # Allocate results array
    mask = np.zeros_like(prediction_bin)
    centroid_list = []

    # # Normalize intensity image (seems to work better than floats with negative values)
    # intensity_image = 65535 * (intensity_image / 2) + 0.5
    # intensity_image = np.clip(intensity_image, 0, 65535).astype(np.uint16)
    #
    # # Check if particles are darker or brighter than background
    # mean_particle_intensity = np.mean(intensity_image, where=(prediction_bin == 1))
    # mean_background_intensity = np.mean(intensity_image, where=(prediction_bin == 0))
    # if mean_background_intensity > mean_particle_intensity:  # particles are dark
    #     intensity_image = 65535 - intensity_image  # invert for weighted centroid calculation

    # Get seeds
    props = measure.regionprops(markers, intensity_image=intensity_image)
    for i in range(len(props)):

        # centroid_weighted = props[i].centroid_weighted
        centroid = props[i].centroid

        # # Sometimes (very rare, weird bug and weighted centroid is even outside the image ...)
        # if np.sqrt(np.sum(np.square(np.array(centroid)-np.array(centroid_weighted)))) > np.sqrt(2):
        #     print('use centroid')
        #     mask[tuple(np.round(centroid).astype(np.uint16))] = True
        # else:
        #     mask[tuple(np.round(centroid_weighted).astype(np.uint16))] = True
        centroid_list.append(centroid)
        mask[tuple(np.round(centroid).astype(np.uint16))] = True

    # Get counts
    counts = int(np.sum(mask))

    return mask, counts, centroid_list
