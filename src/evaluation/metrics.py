import numpy as np
import numpy.typing as npt
from itertools import chain
from skimage import measure
from typing import Any, Tuple, Union

from src.utils.utils import get_particle_ids


def metric_scores(prediction: npt.NDArray[Union[bool, np.ubyte, np.ushort]],
                  ground_truth: npt.NDArray[np.ushort]
                  ) -> Tuple[float, float, float, int, int, int, int, int, int, int]:
    """ Calculate metrics
    
    :param prediction: Particle centroid prediction.
    :param ground_truth: Intensity-coded ground truth area.
    :return: Metrics (precision, recall, f-score, splits, added, false negatives, true positives, false positives,
    number of predicted particles, number of ground truth particles)
    """

    # Assign each predicted particle centroid a unique id
    prediction = measure.label(prediction, connectivity=1, background=0)
    
    # Get the ground truth particle ids and the prediction particle seeds ids
    gt_ids, pred_ids = get_particle_ids(ground_truth), get_particle_ids(prediction)

    # Preallocate lists for used particles and for split particles
    used_ids_pred, used_ids_gt, split_ids_pred, split_ids_gt = [], [], [], []

    # Get particle centroid predictions that lie in a ground truth area, which is intensity-coded
    seeds_in_gt = ground_truth * (prediction > 0)
    seeds_in_gt_hist = np.histogram(seeds_in_gt, bins=range(1, gt_ids[-1] + 2), range=(1, gt_ids[-1] + 1))

    # Get prediction particle ids that match a gt particle (regardless of splits, ...)
    used_ids_pred.append(get_particle_ids(prediction * (seeds_in_gt > 0)))

    # Get gt particles that have (at least one) predicted particle centroid inside
    used_ids_gt.append(get_particle_ids(seeds_in_gt))

    # Find split particles (ids of the multiple predicted particles and of the corresponding gt particle)
    for i, num_particles in enumerate(seeds_in_gt_hist[0]):
        if num_particles > 1:
            split_ids_gt.append(seeds_in_gt_hist[1][i])
            split_ids_pred.append(get_particle_ids(prediction * (seeds_in_gt == seeds_in_gt_hist[1][i])))

    # Count split particles and check for multiple splits within one ground truth particles (the splitting of a gt 
    # particle into three seeds is counted as two splits, ...)
    num_split = 0
    for i in range(len(split_ids_pred)):
        num_split += len(split_ids_pred[i]) - 1

    # Find missing particles (gt particle not marked as used or split)
    num_missing = 0
    ids_gt = list(chain.from_iterable(used_ids_gt)) + split_ids_gt
    for particle_id in gt_ids:
        if particle_id not in ids_gt:
            num_missing += 1

    # Find added particles (predicted particle not marked as used or split)
    num_added = 0
    ids_pred = list(chain.from_iterable(used_ids_pred)) + list(chain.from_iterable(split_ids_pred))
    for particle in pred_ids:
        if particle not in ids_pred:
            num_added += 1

    # Calculate true positives, false positives and false negatives
    tp = len(pred_ids) - num_split - num_added
    fp = num_split + num_added
    fn = num_missing

    # Precision, recall and f_score of the image
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (recall + precision) > 0 else 0

    return precision, recall, f_score, num_split, num_added, fn, tp, fp, len(pred_ids), len(gt_ids)
