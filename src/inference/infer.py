import json
import numpy as np
import omero.model
import os
import pandas as pd
import tifffile as tiff
import torch

from datetime import datetime
from omero.constants import metadata
from omero.gateway import BlitzGateway, MapAnnotationWrapper
from omero.rtypes import rint, rdouble
from pathlib import Path
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot, QCoreApplication
from skimage.transform import rescale

from src.inference.postprocessing import centroid_postprocessing
from src.utils.unets import build_unet, get_weights
from src.utils.data_import import create_roi
from src.utils.utils import zero_pad_model_input, generate_overlay


class InferWorker(QObject):
    """ Worker class for training crop creation """
    finished = pyqtSignal()  # Signal when inference is finished
    text_output = pyqtSignal(str)  # Signal for reporting which file is processed
    progress = pyqtSignal(int)  # Signal for updating the progress bar

    stop_inference = False

    def __init__(self, img_id_list, inference_path, scale_factor, omero_username, omero_password, omero_host,
                 omero_port, group_id, model, device, channel=0, upload=True, overwrite=True, sliding_window=False):
        """

        :param img_id_list: List of omero image ids
        :type img_id_list: list
        :param inference_path: Local results path
        :type inference_path: pathlib Path object
        :param omero_username: OMERO username
        :type omero_username: str
        :param omero_password: OMERO password
        :type omero_password: str
        :param omero_host: OMERO host
        :type omero_host: str
        :param omero_port: OMERO port
        :type omero_port: str
        :param group_id: Current user group id
        :type group_id: int
        :param model: Model to use for inference
        :type model: pathlib Path object
        :param device: Device to use (gpu or cpu)
        :type device: torch device
        :param channel: Channel to process
        :type channel: int
        :param upload: Upload results to OMERO
        :type upload: bool
        :param overwrite: Overwrite results on OMERO
        :type overwrite: bool
        :param sliding_window: Use sliding window for inference (not implemented yet)
        :type sliding_window: bool
        """

        super().__init__()

        self.img_id_list = img_id_list
        self.omero_username = omero_username
        self.omero_password = omero_password
        self.omero_host = omero_host
        self.omero_port = omero_port
        self.conn = None
        self.inference_path = inference_path  # path for local results
        self.channel = channel
        self.scale_factor = scale_factor
        self.upload = upload
        self.overwrite = overwrite
        self.sliding_window = sliding_window
        self.device = device
        self.group_id = group_id
        self.net = None

        # Load model json file to get architecture + filters
        with open(model) as f:
            self.model_settings = json.load(f)
        self.model = model
        self.model_name = "{}: {}".format(model.parent.stem, model.stem)

        # Connect to omero
        self.connect()

    def connect(self):
        """ Connect to OMERO server """
        self.conn = BlitzGateway(self.omero_username, self.omero_password,
                                 host=self.omero_host,
                                 port=self.omero_port,
                                 secure=True)
        self.conn.connect()
        if self.group_id is not None:
            self.conn.setGroupForSession(self.group_id)

    def disconnect(self):
        """ Disconnect from OMERO server"""
        try:
            self.conn.close()
        except:
            pass  # probably already closed / timeout

    def start_inference(self):
        """ Start inference """

        self.text_output.emit('\nInference')

        # Build net
        self.net = build_unet(act_fun=self.model_settings['architecture'][2],
                              pool_method=self.model_settings['architecture'][1],
                              normalization=self.model_settings['architecture'][3],
                              device=self.device,
                              num_gpus=1,  # Only batch size 1 is used at the time, so makes no sense to use more gpus
                              ch_in=1,
                              ch_out=1,
                              filters=self.model_settings['architecture'][4])

        # Load weights
        self.net = get_weights(net=self.net,
                               weights=str(self.model.parent / "{}.pth".format(self.model.stem)),
                               num_gpus=1,
                               device=self.device)

        for i, img_id in enumerate(self.img_id_list):

            # Load image from omero
            try:
                img_ome = self.conn.getObject("Image", img_id)
            except Exception as e:  # probably timeout  --> reconnect and try again
                self.disconnect()
                self.connect()
                img_ome = self.conn.getObject("Image", img_id)

            # When using copy & paste in omero, only links are created and the parent project may be not clear
            parents = img_ome.listParents()
            if len(parents) > 1:
                parent_projects = []
                for pp in parents:
                    if isinstance(pp.getParent(), omero.gateway.ProjectWrapper):
                        parent_projects.append(pp.getParent().getName())
                    else:
                        parent_projects.append(pp.getName())
            else:
                parent_projects = img_ome.getProject().getName()

            if self.upload and not img_ome.canAnnotate():
                self.text_output.emit(f'  Skip {parent_projects}: {img_ome.getName()} (no write permission)')
                continue

            # Check if z stack
            if img_ome.getSizeZ() > 1:
                self.text_output.emit(f'  {parent_projects}: {img_ome.getName()} is z-stack -> '
                                      f'use maximum intensity projection')
                # continue

            # Get image from Omero
            if self.channel + 1 > img_ome.getSizeC():
                self.text_output.emit(f'  Skip {parent_projects}: {img_ome.getName()} (not enough channels found)')
                continue

            # Check if results exist and should not be overwritten
            already_processed = False
            if self.upload:  # Check if annotation is available --> already processed once
                for ann in img_ome.listAnnotations():
                    if ann.OMERO_TYPE == omero.model.MapAnnotationI:
                        keys_values = ann.getValue()
                        for key, value in keys_values:
                            if key in ['inference_model', 'inference_date']:
                                already_processed = True
                if self.overwrite:  # Delete possible annotations
                    roi_service = self.conn.getRoiService()
                    result = roi_service.findByImage(img_ome.getId(), None)
                    roi_ids = [roi.id.val for roi in result.rois if type(roi.getShape(0)) == omero.model.PointI]
                    if roi_ids:
                        try:
                            self.conn.deleteObjects("Roi", roi_ids, wait=True)
                        except:
                            self.text_output.emit(
                                '  Something went wrong during the deletion of already available ROIs')

                    # Delete attachments from label tool and analysis
                    to_delete = []
                    for ann in img_ome.listAnnotations(ns='beadnet.counts.namespace'):
                        to_delete.append(ann.getId())
                    for ann in img_ome.listAnnotations(ns='beadnet.centroids.namespace'):
                        to_delete.append(ann.getId())
                    if to_delete:
                        self.conn.deleteObjects('Annotation', to_delete, wait=True)

            else:  # Check if local results are available
                if isinstance(parent_projects, list):
                    result_path = self.inference_path / parent_projects[0]
                else:
                    result_path = self.inference_path / img_ome.getProject().getName()
                fname = result_path / img_ome.getName()
                if (result_path / f"{fname.stem}_channel{self.channel}.tif").is_file():
                    already_processed = True

            if already_processed and not self.overwrite:
                self.text_output.emit(f'  Skip {parent_projects}: {img_ome.getName()} (already processed and '
                                      f'overwriting not enabled)')
                continue

            if not self.upload:
                if isinstance(parent_projects, list):
                    result_path = self.inference_path / parent_projects[0]
                else:
                    result_path = self.inference_path / img_ome.getProject().getName()
                result_path.mkdir(exist_ok=True)

            # Pre-allocate array for image and results
            img_list, img_upsampled_list = [], []
            count_dict = {'frame': [],
                          'counts': []}
            centroid_dict = {'frame': [],
                             'channel': [],
                             'y': [],
                             'x': [],
                             'y scaled': [],
                             'x scaled': [],
                             'scale factor': []}

            results_array = np.zeros(shape=(img_ome.getSizeT(),
                                            img_ome.getSizeY() * self.scale_factor,
                                            img_ome.getSizeX() * self.scale_factor),
                                     dtype=np.uint8)

            # Go through frames
            for frame in range(img_ome.getSizeT()):

                # Check for stop signal
                QCoreApplication.processEvents()  # Update to get stop signal
                if self.stop_inference:
                    self.text_output.emit("Stop inference due to user interaction.")
                    self.progress.emit(0)
                    self.disconnect()
                    self.finished.emit()
                    return

                # Get image from Omero (maximum intensity projection for z-stacks)
                slice_list = []
                if img_ome.getSizeZ() > 1:
                    for slice_idx in range(img_ome.getSizeZ()):
                        slice_list.append(img_ome.getPrimaryPixels().getPlane(slice_idx,
                                                                              self.channel,
                                                                              0))
                    img = np.array(slice_list)
                    img = np.max(img, axis=0)
                else:
                    img = img_ome.getPrimaryPixels().getPlane(0, self.channel, frame)
                img_list.append(np.copy(img))

                # Upsampling
                if self.scale_factor > 1:
                    img = rescale(img, self.scale_factor, preserve_range=True, order=2).astype(img.dtype)
                    img_upsampled_list.append(np.copy(img))

                # Get frame_min and frame_max before padding/cropping
                img_min, img_max, img_mean, img_std = np.min(img), np.max(img), np.mean(img), np.std(img)

                # Zero padding
                img, pads = zero_pad_model_input(img, pad_val=img_min)

                # Predict crop
                prediction, counts, centroids = self.inference(img, min_val=img_min, max_val=img_max, pads=pads)

                if counts == 0:
                    continue

                # Append results
                count_dict['frame'].append(frame)
                count_dict['counts'].append(counts)
                for centroid in centroids:
                    centroid_dict['frame'].append(frame)
                    centroid_dict['channel']. append(self.channel)
                    centroid_dict['y'].append(centroid[0] / self.scale_factor)
                    centroid_dict['x'].append(centroid[1] / self.scale_factor)
                    centroid_dict['y scaled'].append(centroid[0])
                    centroid_dict['x scaled'].append(centroid[1])
                    centroid_dict['scale factor'].append(self.scale_factor)

                # Fill results array
                results_array[frame] = prediction

                # Create Point ROIs for each cell and upload segmentation
                if self.upload:
                    update_service = self.conn.getUpdateService()
                    img_ome = self.conn.getObject("Image", img_ome.getId())  # Reload needed
                    mask_point_roi_list = []
                    if np.max(prediction) > 0:
                        try:
                            for j in range(len(centroids)):
                                if self.scale_factor > 1:
                                    centroid = np.array(centroids[j]) / self.scale_factor
                                else:
                                    centroid = centroids[j]
                                mask_point_roi = omero.model.PointI()
                                mask_point_roi.x = rdouble(centroid[1])
                                mask_point_roi.y = rdouble(centroid[0])
                                mask_point_roi.theZ = rint(0)  # append mip-prediction always to first slice
                                mask_point_roi.theC = rint(self.channel)
                                mask_point_roi.theT = rint(frame)
                                mask_point_roi.strokeColor = rint(
                                    int.from_bytes([255, 255, 0, 255], byteorder='big', signed=True))
                                mask_point_roi_list.append(mask_point_roi)
                            update_service.saveAndReturnObject(create_roi(img_ome, mask_point_roi_list))
                        except:
                            pass

                # Upload metadata
                if frame == 0 and self.upload:
                    now = datetime.now()
                    key_value_data = [['inference_model', self.model_name],
                                      ['inference_date', now.strftime("%m/%d/%Y, %H:%M:%S")]]
                    for ann in img_ome.listAnnotations():
                        if ann.OMERO_TYPE == omero.model.MapAnnotationI:
                            keys_values = ann.getValue()
                            for key, value in keys_values:
                                if key not in ['inference_model', 'inference_date', 'last_modification']:
                                    key_value_data.append([key, value])
                            if ann.canEdit():
                                self.conn.deleteObjects("Annotation", [ann.getId()], wait=True)
                            else:
                                self.text_output.emit(
                                    f'  Problems with deleting annotations (probably from another user), redundant'
                                    f' results possible. Please check on OMERO.web.')
                    map_ann = MapAnnotationWrapper(self.conn)
                    map_ann.setNs(
                        metadata.NSCLIENTMAPANNOTATION)  # Use 'client' namespace to allow editing in Insight & web
                    map_ann.setValue(key_value_data)
                    map_ann.save()
                    img_ome.linkAnnotation(map_ann)

                # Update progress bar
                self.progress.emit(int(100 * ((i + (frame + 1) / img_ome.getSizeT()) / len(self.img_id_list))))

            img, img_upsampled = np.squeeze(np.array(img_list)), np.squeeze(np.array(img_upsampled_list))
            results_array = np.squeeze(results_array)
            counts_df = pd.DataFrame(count_dict)
            centroids_df = pd.DataFrame(centroid_dict)
            if not self.upload:  # Save locally
                fname = result_path / img_ome.getName()
                tiff.imsave(str(result_path / f"{fname.stem}_ch{self.channel}_img.tif"), img, compress=1)
                if self.scale_factor > 1:
                    tiff.imsave(str(result_path / f"{fname.stem}_ch{self.channel}_img_upsampled.tif"), img_upsampled, compress=1)
                    overlay = generate_overlay(img_upsampled, results_array)
                else:
                    overlay = generate_overlay(img, results_array)
                tiff.imsave(str(result_path / f"{fname.stem}_ch{self.channel}_mask.tif"), results_array, compress=1)
                tiff.imsave(str(result_path / f"{fname.stem}_ch{self.channel}_overlay.tif"), overlay, compress=1)
                centroids_df.to_csv(result_path / f"{fname.stem}_ch{self.channel}_centroids.csv", header=True, index=False)
                counts_df.to_csv(result_path / f"{fname.stem}_ch{self.channel}_counts.csv", header=True, index=False)
            else:  # attach lists to omero image
                # Save temporarily (needed for upload)
                fname = (Path.cwd() / img_ome.getName()).stem
                centroids_df.to_csv(Path.cwd() / f"{fname}_ch{self.channel}_centroids.csv", header=True, index=False)
                counts_df.to_csv(Path.cwd() / f"{fname}_ch{self.channel}_counts.csv", header=True, index=False)

                # Delete old results if available
                to_delete = []
                for ann in img_ome.listAnnotations(ns='beadnet.counts.namespace'):
                    to_delete.append(ann.getId())
                for ann in img_ome.listAnnotations(ns='beadnet.centroids.namespace'):
                    to_delete.append(ann.getId())
                if to_delete:
                    self.conn.deleteObjects('Annotation', to_delete, wait=True)

                # Upload new results
                file_ann = self.conn.createFileAnnfromLocalFile(str(Path.cwd() / f"{fname}_ch{self.channel}_centroids.csv"),
                                                                mimetype='text/csv',
                                                                ns='beadnet.centroids.namespace',
                                                                desc='BeadNet counts')
                img_ome.linkAnnotation(file_ann)
                file_ann = self.conn.createFileAnnfromLocalFile(str(Path.cwd() / f"{fname}_ch{self.channel}_counts.csv"),
                                                                mimetype='text/csv',
                                                                ns='beadnet.counts.namespace',
                                                                desc='BeadNet detection results')
                img_ome.linkAnnotation(file_ann)

                # Remove temp file
                os.remove(str(Path.cwd() / f"{fname}_ch{self.channel}_centroids.csv"))
                os.remove(str(Path.cwd() / f"{fname}_ch{self.channel}_counts.csv"))

        self.disconnect()
        self.progress.emit(100)
        self.finished.emit()

    def inference(self, img, min_val, max_val, pads):
        """ Prediction of crops for pre-labeling

        :param img: Crop to predict
        :type img:
        :param min_val: Minimum image/frame value for normalization
        :type min_val: int
        :param max_val: Maximum image/frame value for normalization
        :type max_val: int
        :return: prediction
        :param pads: Number of padded zeros in each dimension (need to be removed after inference)
        :type pads: list
        """

        self.net.eval()
        torch.set_grad_enabled(False)

        # Normalize crop and convert to tensor / img_batch
        img_batch = 2 * (img.astype(np.float32) - min_val) / (max_val - min_val) - 1
        img_batch = torch.from_numpy(img_batch[None, None, :, :]).to(torch.float)
        img_batch = img_batch.to(self.device)

        # Prediction
        try:
            prediction_batch = self.net(img_batch)
        except RuntimeError:
            prediction = (np.zeros_like(img, dtype=np.uint16)[pads[0]:, pads[1]:], 0, [])
            self.text_output.emit('RuntimeError during inference (maybe not enough ram/vram?)')
        else:
            prediction_batch = torch.sigmoid(prediction_batch)
            prediction_batch = prediction_batch[:, :, pads[0]:, pads[1]:].cpu().numpy()
            prediction_batch = np.transpose(prediction_batch[0], (1, 2, 0))
            prediction = centroid_postprocessing(np.squeeze(prediction_batch))

        return prediction

    @pyqtSlot()
    def inference_finished(self):
        """ Close connection and send finished signal """
        self.disconnect()
        self.finished.emit()

    @pyqtSlot()
    def stop_inference_process(self):
        """ Set internal stop state to True

        :return: None
        """
        self.stop_inference = True
