import numpy as np
import omero.model
import os
import pandas as pd
import tifffile as tiff

from omero.gateway import BlitzGateway
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot, QCoreApplication
from skimage.transform import rescale

from src.utils.utils import generate_overlay


class ResultExportWorker(QObject):
    """ Worker class for dataset import """
    finished = pyqtSignal()  # Signal when import is finished
    progress = pyqtSignal(int)  # Signal for updating the progress bar
    text_output = pyqtSignal(str)  # Signal for possible exceptions, e.g., user interaction to stop export
    stop_export = False

    def __init__(self, img_id_list, inference_path, channel, omero_username, omero_password,  omero_host, omero_port,
                 group_id):
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
        """

        super().__init__()

        self.img_id_list = img_id_list
        self.omero_username = omero_username
        self.omero_password = omero_password
        self.omero_host = omero_host
        self.omero_port = omero_port
        self.group_id = group_id
        self.channel = channel
        self.conn = None
        self.inference_path = inference_path  # path for local results
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

    def export_data(self):
        """ Export results from Omero """

        self.text_output.emit('\nExport data and results')

        for i, img_id in enumerate(self.img_id_list):

            QCoreApplication.processEvents()  # Update to get stop signal
            if self.stop_export:
                self.text_output.emit("Stop result export due to user interaction.")
                self.progress.emit(0)
                self.disconnect()
                self.finished.emit()
                return

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

            # Get path and filename
            if isinstance(parent_projects, list):
                result_path = self.inference_path / parent_projects[0]
            else:
                result_path = self.inference_path / img_ome.getProject().getName()
            result_path.mkdir(exist_ok=True)
            fname = result_path / img_ome.getName()

            self.progress.emit(int(100 * ((i + 1 / 10) / len(self.img_id_list))))

            # Check if Point ROIs are available
            roi_service = self.conn.getRoiService()
            point_rois = roi_service.findByImage(img_ome.getId(), None)
            counts_df, centroids_df = None, None
            for point_roi in point_rois.rois:
                for s in point_roi.copyShapes():
                    if type(s) == omero.model.PointI:
                        # Get and save analysis results (counts ...)
                        for ann in img_ome.listAnnotations(ns='beadnet.counts.namespace'):
                            with open(result_path / f"{fname.stem}_ch{self.channel}_counts.csv", 'wb') as outfile:
                                for chunk in ann.getFileInChunks():
                                    outfile.write(chunk)
                            counts_df = pd.read_csv(result_path / f"{fname.stem}_ch{self.channel}_counts.csv")
                        for ann in img_ome.listAnnotations(ns='beadnet.centroids.namespace'):
                            with open(result_path / f"{fname.stem}_ch{self.channel}_centroids.csv", 'wb') as outfile:
                                for chunk in ann.getFileInChunks():
                                    outfile.write(chunk)
                            centroids_df = pd.read_csv(result_path / f"{fname.stem}_ch{self.channel}_centroids.csv")
                        break  # break second for loop?
                if counts_df is not None:
                    break
            if counts_df is None or centroids_df is None:
                self.text_output.emit(f'  Skip {parent_projects}: {img_ome.getName()} (no detection results found)')
                continue
            if centroids_df['channel'][0] != self.channel:
                self.text_output.emit(f'  Skip {parent_projects}: {img_ome.getName()} (no detection results for '
                                      f'selected channel found)')
                os.remove(str(result_path / f"{fname.stem}_ch{self.channel}_centroids.csv"))
                os.remove(str(result_path / f"{fname.stem}_ch{self.channel}_counts.csv"))
                continue

            self.progress.emit(int(100 * ((i + 1 / 4) / len(self.img_id_list))))

            QCoreApplication.processEvents()  # Update to get stop signal
            if self.stop_export:
                self.text_output.emit("Stop result export due to user interaction.")
                self.progress.emit(0)
                self.disconnect()
                self.finished.emit()
                return

            # Get image
            img = np.zeros(shape=(img_ome.getSizeT(), img_ome.getSizeZ(), img_ome.getSizeY(), img_ome.getSizeX(), 1),
                           dtype=img_ome.getPixelsType())
            zct_list = []
            for z in range(img_ome.getSizeZ()):  # all slices
                # for c in range(img_ome.getSizeC()):  # all channels
                for t in range(img_ome.getSizeT()):  # all time-points
                    zct_list.append((z, self.channel, t))
            for h, plane in enumerate(img_ome.getPrimaryPixels().getPlanes(zct_list)):
                img[zct_list[h][2], zct_list[h][0], :, :, zct_list[h][1]] = plane
            img = np.squeeze(img)
            self.progress.emit(int(100 * ((i + 2 / 4) / len(self.img_id_list))))

            QCoreApplication.processEvents()  # Update to get stop signal
            if self.stop_export:
                self.text_output.emit("Stop result export due to user interaction.")
                self.progress.emit(0)
                self.disconnect()
                self.finished.emit()
                return

            if img_ome.getSizeZ() > 1:  # Use maximum intensity projection
                if img_ome.getSizeT() == 1:
                    img = np.max(img, axis=0)
                else:
                    img = np.max(img, axis=1)

            self.progress.emit(int(100 * ((i + 3 / 4) / len(self.img_id_list))))

            # Upsampling if it has been applied
            scale_factor = centroids_df['scale factor'][0]
            if scale_factor > 1:
                if len(img.shape) == 2:
                    img_upsampled = rescale(img, scale_factor, preserve_range=True, order=2).astype(img.dtype)
                else:  # time series
                    img_upsampled = rescale(img, (1, scale_factor, scale_factor), preserve_range=True, order=2).astype(img.dtype)

            # Fill mask
            mask = np.zeros(shape=(img_ome.getSizeT(),
                                   img_ome.getSizeY() * scale_factor,
                                   img_ome.getSizeX() * scale_factor),
                            dtype=np.uint8)
            for _, row in centroids_df[['frame', 'x scaled', 'y scaled']].iterrows():  # scaled works also for scale factor = 1
                mask[int(np.round(row['frame'])),
                     int(np.round(row['y scaled'])),
                     int(np.round(row['x scaled']))] = 1
            mask = np.squeeze(mask)

            # Generate overlay
            if scale_factor > 1:
                img_upsampled = np.squeeze(img_upsampled)
                overlay = generate_overlay(img_upsampled, mask)
            else:
                overlay = generate_overlay(img, mask)

            # Save image, mask and label
            tiff.imwrite(str(result_path / f"{fname.stem}_ch{self.channel}_img.tif"), img, compress=1)
            if scale_factor > 1:
                tiff.imwrite(str(result_path / f"{fname.stem}_ch{self.channel}_img_upsampled.tif"), img_upsampled, compress=1)
            tiff.imsave(str(result_path / f"{fname.stem}_ch{self.channel}_mask.tif"), mask, compress=1)
            tiff.imsave(str(result_path / f"{fname.stem}_ch{self.channel}_overlay.tif"), overlay, compress=1)

            self.progress.emit(int(100 * (i+1) / len(self.img_id_list)))

        self.disconnect()
        self.progress.emit(100)
        self.finished.emit()

    @pyqtSlot()
    def stop_export_process(self):
        """ Set internal export stop state to True

        :return: None
        """
        self.stop_export = True
