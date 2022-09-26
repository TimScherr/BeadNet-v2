import json
import numpy as np
import numpy.typing as npt

from scipy.ndimage import binary_dilation
from scipy.ndimage.morphology import generate_binary_structure


def plane_gen(img):
    """generator will yield planes"""
    for p in [img]:
        yield p


def get_particle_ids(img):
    """ Get particle ids in intensity-coded label image.

    :param img: Intensity-coded nuclei image.
        :type:
    :return: List of nucleus ids.
    """

    values = np.unique(img)
    values = values[values > 0]

    return values


def generate_overlay(img, seeds):
    """ Image-Particle-seeds overlay.

    :param img: Image.
    :type img:
    :param seeds: Particle seeds.
    :type seeds:
    :return: Overlay (img: gray, seeds: red)
    """

    # Normalize image for overlay:
    img = np.clip(255 * img.astype(np.float32) / img.max(), 0, 255).astype(np.uint8)

    if len(img.shape) == 2:
        markers = binary_dilation(seeds, generate_binary_structure(2, 1)) > 0
        markers = np.tile(markers[..., None], (1, 1, 3))
        markers[:, :, 1], markers[:, :, 2] = 0, 0
        overlay = np.tile(img[..., None], (1, 1, 3))
        overlay[markers] = 255
        overlay[markers[:, :, [2, 0, 1]]] = 0
        overlay[markers[:, :, [2, 1, 0]]] = 0
    else:
        overlay = np.tile(img[..., None], (1, 1, 1, 3))
        markers = np.copy(seeds)
        for frame in range(len(markers)):
            markers[frame] = binary_dilation(markers[frame], generate_binary_structure(2, 1)) > 0
        markers = np.tile(markers[..., None] > 0, (1, 1, 1, 3))
        markers[..., 1], markers[..., 2] = 0, 0
        overlay[markers] = 255
        overlay[markers[..., [2, 0, 1]]] = 0
        overlay[markers[..., [2, 1, 0]]] = 0

    return overlay


def border_correction(particle_prediction: npt.NDArray[np.ushort], ground_truth: npt.NDArray[np.ushort],
                      border_size: int = 6) -> npt.NDArray[np.ushort]:
    """ Border correction for evaluation of crops

    :param particle_prediction: Particle centroid prediction.
    :param border_size: Border size.
    :param ground_truth: Particle centroid area ground truth.
    :return: Border corrected particle centroid prediction.
    """

    # Get roi (inner image + where particle annotations exist)
    roi = ground_truth.copy() > 0
    roi[border_size:roi.shape[0] - border_size, border_size:roi.shape[1] - border_size] = 1
    # Apply border correction
    particle_prediction = particle_prediction * roi

    return particle_prediction


def min_max_normalization(img, min_value=None, max_value=None):
    """ Minimum maximum normalization.

    :param img: Image (uint8, uint16 or int)
        :type img:
    :param min_value: minimum value for normalization, values below are clipped.
        :type min_value: int
    :param max_value: maximum value for normalization, values above are clipped.
        :type max_value: int
    :return: Normalized image (float32)
    """

    if max_value is None:
        max_value = img.max()

    if min_value is None:
        min_value = img.min()

    # Clip image to filter hot and cold pixels
    img = np.clip(img, min_value, max_value)

    # Apply min-max-normalization
    img = 2 * (img.astype(np.float32) - min_value) / (max_value - min_value) - 1

    return img.astype(np.float32)


def unique_path(directory, name_pattern):
    """ Get unique file name to save trained model.

    :param directory: Path to the model directory
        :type directory: pathlib path object.
    :param name_pattern: Pattern for the file name
        :type name_pattern: str
    :return:
    """
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path


def write_train_info(configs, path):
    """ Write training configurations into a json file.

    :param configs: Dictionary with configurations of the training process.
        :type configs: dict
    :param path: path to the directory to store the json file.
        :type path: pathlib Path object
    :return: None
    """

    with open(path / (configs['run_name'] + '.json'), 'w', encoding='utf-8') as outfile:
        json.dump(configs, outfile, ensure_ascii=False, indent=2)

    return None


def write_inference_results(results, path):
    """ Write inference results (number of beads) into a json file.

    :param results: Inference results.
        :type results: dict
    :param path: Result path
        :type path: pathlib path object.
    :return: None
    """

    with open(path / 'results.json', 'w', encoding='utf-8') as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=2)


def zero_pad_model_input(img, pad_val=0):
    """ Zero-pad model input to get for the model needed sizes (more intelligent padding ways could easily be
        implemented but there are sometimes cudnn errors with image sizes which work on cpu ...).

    :param img: Model input image.
        :type:
    :param pad_val: Value to pad.
        :type pad_val: int.

    :return: (zero-)padded img, [0s padded in y-direction, 0s padded in x-direction]
    """

    # Tested shapes
    tested_img_shapes = [64, 128, 256, 320, 512, 768, 1024, 1280, 1408, 1600, 1920, 2048, 2240, 2560, 3200, 4096,
                         4480, 6080, 8192]

    if len(img.shape) == 3:  # 3D image (z-dimension needs no pads)
        img = np.transpose(img, (2, 1, 0))

    # More effective padding (but may lead to cuda errors)
    # y_pads = int(np.ceil(img.shape[0] / 64) * 64) - img.shape[0]
    # x_pads = int(np.ceil(img.shape[1] / 64) * 64) - img.shape[1]

    pads = []
    for i in range(2):
        for tested_img_shape in tested_img_shapes:
            if img.shape[i] <= tested_img_shape:
                pads.append(tested_img_shape - img.shape[i])
                break

    if not pads:
        raise Exception('Image too big to pad. Use sliding windows')

    if len(img.shape) == 3:  # 3D image
        img = np.pad(img, ((pads[0], 0), (pads[1], 0), (0, 0)), mode='constant', constant_values=pad_val)
        img = np.transpose(img, (2, 1, 0))
    else:
        img = np.pad(img, ((pads[0], 0), (pads[1], 0)), mode='constant', constant_values=pad_val)

    return img, [pads[0], pads[1]]
