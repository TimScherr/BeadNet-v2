import gc
import json
import hashlib
import shutil

import numpy as np
import os
import pandas as pd
import tifffile as tiff
import torch
import zipfile

from multiprocessing import cpu_count
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot, QCoreApplication
from scipy.ndimage import binary_dilation
from scipy.ndimage.morphology import generate_binary_structure
from skimage import measure
from skimage.morphology import disk
from skimage.segmentation import watershed

from src.utils.unets import build_unet, get_weights
from src.inference.inference_dataset import InferenceDataset, pre_processing_transforms
from src.inference.postprocessing import centroid_postprocessing
from src.evaluation.metrics import metric_scores
from src.utils.utils import border_correction


class EvalWorker(QObject):
    """ Worker class for model evaluation """
    finished = pyqtSignal()  # Signal when import is finished
    progress = pyqtSignal(int)  # Signal for updating the progress bar
    text_output = pyqtSignal(str)  # Signal for possible exceptions, e.g., user interaction to stop export
    stop_evaluation = False  # Stop evaluation process
    is_evaluating = False  # State of evaluation process

    def start_evaluation(self, path_data, path_results, gt_radius, models, batch_size, device, num_gpus, save_raw_pred,
                         start_message=''):
        """ Start evaluation process

        :param path_data: Path of the training set (contains dirs 'train', and 'val').
        :type path_data: pathlib Path object
        :param path_results: Path to save the evaluation results into.
        :type path_results: pathlib Path object
        :param models: List with paths of the models to evaluate.
        :type models: list
        :param batch_size: Batch size.
        :type batch_size: int
        :param device: Device to use (gpu or cpu).
        :type device: torch device
        :param num_gpus: Number of gpus to use.
        :type num_gpus: int
        :param save_raw_pred: Save also the raw cnn outputs.
        :type save_raw_pred: bool
        :param start_message: Message to print at start of evaluation.
        :type start_message: str
        :return: None
        """

        # Check if export has been stopped (folders are deleted)
        if not path_data.is_dir():
            self.progress.emit(0)
            self.finished.emit()
            return

        # Check if enough test images are available:
        if len(list((path_data / 'test').glob('mask*'))) < 2:
            self.text_output.emit('Not enough test images found. At least 2 are needed (better more)')
            self.progress.emit(0)
            self.finished.emit()
            return

        self.is_evaluating = True

        self.text_output.emit(start_message)

        trainset_scores = {'model': [],
                           'ground truth radius': [],
                           'average precision': [],
                           'average recall': [],
                           'average f-score': [],
                           'count rate': [],
                           'true positive rate': [],
                           'split rate': [],
                           'miss rate': [],
                           'add rate': [],
                           'test set version': []}

        # Eval multiple models
        for i, model in enumerate(models):

            path_results_model = path_results / f"{model.parent.stem}_{model.stem}_radius_{gt_radius}"

            # Make dirs / clean dirs
            if not path_results.is_dir():
                path_results.mkdir()
            if not path_results_model.is_dir():
                path_results_model.mkdir()
            else:
                shutil.rmtree(path_results_model)
                path_results_model.mkdir()

            # Look for stop signal
            QCoreApplication.processEvents()  # Update to get stop signal
            if self.stop_evaluation:
                if self.is_evaluating:  # Stop event happened outside evaluation (print stop message)
                    self.text_output.emit("Stop evaluation due to user interaction.")
                break

            # Load model json file to get architecture + filters
            with open(model.parent / "{}.json".format(model.stem)) as f:
                model_settings = json.load(f)

            # Build CNN
            net = build_unet(act_fun=model_settings['architecture'][2],
                             pool_method=model_settings['architecture'][1],
                             normalization=model_settings['architecture'][3],
                             device=device,
                             num_gpus=num_gpus,
                             ch_in=1,
                             ch_out=1,
                             filters=model_settings['architecture'][4])

            # Load weights
            net = get_weights(net=net, weights=str(model), num_gpus=num_gpus, device=device)

            # Get test set (new dataset needed)
            test_dataset = InferenceDataset(data_dir=path_data / 'test',
                                            transform=pre_processing_transforms(apply_clahe=False, scale_factor=1))

            # Inference (save results for multiple thresholds at once)
            try:
                self.inference(net=net, dataset=test_dataset, batch_size=batch_size, device=device,
                               path_model=path_results_model, save_raw=save_raw_pred,
                               eval_progress=(0.5/len(models), i/len(models)))
            except:
                text = "Error: probably VRAM/RAM problem ..."
                self.text_output.emit(text)
                self.text_output.emit('Stop evaluation')
                self.text_output.emit(text)
                self.finished.emit()
                return

            # Clear memory
            del net
            gc.collect()

            # Calculate scores (keep only best threshold results)
            results = self.calc_scores(prediction_path=path_results_model, test_set_path=path_data / 'test',
                                       gt_radius=gt_radius)

            if results:  # check if operation was aborted
                trainset_scores['model'].append("{}: {}".format(model.parent.stem, model.stem))
                trainset_scores['ground truth radius'].append(results[8])
                trainset_scores['average precision'].append(results[0])
                trainset_scores['average recall'].append(results[1])
                trainset_scores['average f-score'].append(results[2])
                trainset_scores['count rate'].append(results[3])
                trainset_scores['split rate'].append(results[4])
                trainset_scores['true positive rate'].append(results[5])
                trainset_scores['miss rate'].append(results[6])
                trainset_scores['add rate'].append(results[7])
                trainset_scores['test set version'].append(results[9])

                # zip test set and save into evaluation folder
                with zipfile.ZipFile(path_results_model / 'test_set.zip', 'w') as z:
                    z.write(path_data, arcname=path_data.stem, compress_type=zipfile.ZIP_DEFLATED)
                    z.write(path_data / 'test', arcname=os.path.join(path_data.stem, 'test'),
                            compress_type=zipfile.ZIP_DEFLATED)
                    for file in (path_data / 'test').glob('*'):
                        z.write(file, arcname=os.path.join(path_data.stem, 'test', file.name),
                                compress_type=zipfile.ZIP_DEFLATED)

            # Update progress bar
            self.progress.emit(int(100 * (i + 1) / len(models)))

        if not self.stop_evaluation:
            # Convert to pandas dataframe
            trainset_scores_df = pd.DataFrame(trainset_scores)
            # Get existing scores
            if (path_results.parent / '{}.csv'.format(path_results.stem)).is_file():
                trainset_scores_old_df = pd.read_csv(path_results.parent / '{}.csv'.format(path_results.stem))
                # Delete evaluation scores on old test set (hash differs)
                trainset_scores_old_df = trainset_scores_old_df[
                    trainset_scores_old_df['test set version'] == trainset_scores_df.iloc[0]['test set version']]
                # Combine old and new scores without duplicates
                trainset_scores_df = trainset_scores_df.append(trainset_scores_old_df)
                # Delete possible duplicate keys due to rounding errors ...
                trainset_scores_df = trainset_scores_df.drop_duplicates(subset=['model', 'ground truth radius'])
            trainset_scores_df = trainset_scores_df.sort_values(by=['model', 'ground truth radius'])
            trainset_scores_df.to_csv(path_results.parent / '{}.csv'.format(path_results.stem),
                                      float_format='%.4f',
                                      header=True,
                                      index=False)
            self.progress.emit(100)
        self.finished.emit()

        return

    @pyqtSlot()
    def stop_evaluation_process(self):
        """ Set internal evaluation stop state to True

        :return: None
        """
        self.stop_evaluation = True

    def calc_scores(self, prediction_path, test_set_path, gt_radius=2):
        """ Calculate metrics (aggregated Jaccard index).

        :param prediction_path: Path to the predictions.
        :type prediction_path: pathlib Path object.
        :param test_set_path: Path to the ground truths.
        :type test_set_path: pathlib Path object.
        :return: [mean score, std score] for boundary method, [mean score, std_score, [th_cell, th_seed]] for distance
        """

        precision, recall, f_score, split, added, fn, tp, fp = [], [], [], [], [], [], [], []
        pred_count, gt_count = [], []
        file_names, radii = [], []

        pred_ids = prediction_path.glob('mask*.tif')
        for pred_id in pred_ids:

            # Look for stop signal
            QCoreApplication.processEvents()  # Update to get stop signal
            if self.stop_evaluation:
                if self.is_evaluating:  # Stop event happened outside evaluation (print stop message)
                    self.text_output.emit("Stop metric calculation.")
                return None

            particle_prediction = tiff.imread(str(pred_id)) > 0
            ground_truth = tiff.imread(str(test_set_path / pred_id.name)) > 0

            # Get ground truth area
            ground_truth_markers = measure.label(ground_truth, connectivity=1, background=0)
            if gt_radius == 1:
                ground_truth_area = binary_dilation(ground_truth, generate_binary_structure(2, 1))
            elif gt_radius == 1.5:
                ground_truth_area = binary_dilation(ground_truth, generate_binary_structure(2, 2))
            elif gt_radius == 2:
                ground_truth_area = binary_dilation(ground_truth, disk(2))
            elif gt_radius == 3:
                ground_truth_area = binary_dilation(ground_truth, disk(3))
            else:
                raise Exception('Radius not supported')
            ground_truth_area = watershed(image=ground_truth_area, markers=ground_truth_markers, mask=ground_truth_area,
                                          watershed_line=False).astype(np.uint16)

            # Apply border correction
            particle_prediction = border_correction(particle_prediction, ground_truth_area)

            # Save overlay: pred (white dot) in ground truth area (gray)
            pred_gt_overlay = np.maximum(ground_truth_area > 0, 2 * particle_prediction).astype(np.uint8)

            tiff.imwrite(str(pred_id.parent / f"overlay{pred_id.name.split('mask')[-1]}"), pred_gt_overlay)
            # tiff.imwrite(str(pred_id.parent / f"ground_truth{pred_id.name.split('mask')[-1]}"), ground_truth_area.astype(np.uint16))

            scores = metric_scores(particle_prediction, ground_truth_area)

            # Append metrics to corresponding lists
            precision.append(scores[0])
            recall.append(scores[1])
            f_score.append(scores[2])
            split.append(scores[3])
            added.append(scores[4])
            fn.append(scores[5])
            tp.append(scores[6])
            fp.append(scores[7])
            pred_count.append(scores[8])
            gt_count.append(scores[9])
            radii.append(gt_radius)
            file_names.append(pred_id.stem)

        # Save scores
        results_df = pd.DataFrame({'test image': file_names,
                                   'precision': precision,
                                   'recall': recall,
                                   'f-score': f_score,
                                   'true positives': tp,
                                   'false positives': fp,
                                   'false negatives': fn,
                                   'splits': split,
                                   'predicted particles': pred_count,
                                   'ground truth particles': gt_count,
                                   'ground truth radius': gt_radius})
        results_df = results_df.sort_values(by=['test image'])
        results_df.to_csv(prediction_path / "scores.csv", float_format='%.6f', header=True, index=False)

        # Calculate average scores
        precision = np.mean(np.array(precision))
        recall = np.mean(np.array(recall))
        f_score = 2 * precision * recall / (precision + recall) if (recall + precision) > 0 else 0
        gt_count = np.sum(np.array(gt_count))
        detection_rate = np.sum(np.array(pred_count)) / gt_count
        split_rate = np.sum(np.array(split)) / gt_count
        tp_rate = np.sum(np.array(tp)) / gt_count
        miss_rate = np.sum(np.array(fn)) / gt_count
        add_rate = np.sum(np.array(fp)) / gt_count

        return precision, recall, f_score, detection_rate, split_rate, tp_rate, miss_rate, add_rate, \
               gt_radius, hashlib.sha1(str(file_names).encode("UTF-8")).hexdigest()[:10]

    def inference(self, net, dataset, batch_size, device, path_model, save_raw, eval_progress):
        """ Train the model.
        :param net: Model/Network to use for inference.
        :type net:
        :param dataset: Pytorch dataset.
        :type dataset: torch dataset.
        :param batch_size: Batch size.
        :type batch_size: int
        :param device: Device to use (gpu or cpu).
        :type device: torch device
        :param path_model: Path to save the results of the selected model into.
        :type path_model: pathlib Path object
        :param save_raw: Save also raw cnn outputs.
        :type save_raw: bool.
        :param eval_progress: Progress of the evaluation (needed to update the progress bar properly)
        :type eval_progress: tuple
        :return: None
        """

        # Data loader for training and validation set
        if device.type == "cpu":
            num_workers = 0
        else:
            try:
                num_workers = cpu_count() // 2
            except AttributeError:
                num_workers = 4
        num_workers = np.minimum(num_workers, 16)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                 num_workers=num_workers)

        net.eval()
        torch.set_grad_enabled(False)

        # Predict images (iterate over images/files)
        for i, sample in enumerate(dataloader):

            if i % 5 == 0:  # Check from time to time if evaluation should be stopped
                QCoreApplication.processEvents()
                if self.stop_evaluation:
                    self.text_output.emit("Stop evaluation due to user interaction.")
                    self.is_evaluating = False
                    return

            img_batch, ids_batch, pad_batch, img_size = sample
            img_batch = img_batch.to(device)

            if batch_size > 1:  # all images in a batch have same dimensions and pads
                pad_batch = [pad_batch[i][0] for i in range(len(pad_batch))]

            # Prediction
            prediction_batch = net(img_batch)
            prediction_batch = torch.sigmoid(prediction_batch)
            # Get rid of pads
            prediction_batch = prediction_batch[..., pad_batch[0]:, pad_batch[1]:].cpu().numpy()
            prediction_batch = np.transpose(prediction_batch, (0, 2, 3, 1))
            img_batch = img_batch[..., pad_batch[0]:, pad_batch[1]:].cpu().numpy()
            img_batch = np.transpose(img_batch, (0, 2, 3, 1))

            # Go through predicted batch and apply post-processing (not parallelized)
            for h in range(len(img_batch)):

                file_id = ids_batch[h].split('img')[-1]

                if save_raw:
                    tiff.imwrite(str(path_model / "raw{}.tif".format(file_id)), prediction_batch[h])

                prediction, _, _ = centroid_postprocessing(np.squeeze(prediction_batch[h]), np.squeeze(img_batch[h]))

                tiff.imwrite(str(path_model / "mask{}.tif".format(file_id)), prediction.astype(np.uint8))

            # Update progress bar
            self.progress.emit(int(100 * (eval_progress[0] * (i + 1) * batch_size / len(dataset) + eval_progress[1])))

        return
