import numpy as np
import omero.model
import pandas as pd
import tifffile as tiff

from omero.gateway import BlitzGateway
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot, QCoreApplication
from shutil import rmtree
from src.utils.utils import generate_overlay


class DataExportWorker(QObject):
    """ Worker class for dataset import """
    finished = pyqtSignal()  # Signal when import is finished
    progress = pyqtSignal(int)  # Signal for updating the progress bar
    text_output = pyqtSignal(str)  # Signal for possible exceptions, e.g., user interaction to stop export
    stop_export = False

    def export_data(self, trainset_id, train_path, omero_username, omero_password,  omero_host, omero_port, group_id):
        """ Import data to Omero

        :param trainset_id: id of the Omero dataset to import the data into
        :type trainset_id
        :param train_path: Local path to save training data info
        :type train_path: pathlib Path object
        :param omero_username: Omero username
        :type omero_username: str
        :param omero_password: Omero password
        :type omero_password: str
        :param omero_host: Omero host
        :type omero_host: str
        :param omero_port: Omero port
        :type omero_port: str
        :param group_id: Current user group id
        :type group_id: int
        :return: None
        """

        # Connect to Omero
        conn = BlitzGateway(omero_username, omero_password, host=omero_host, port=omero_port, secure=True)
        conn.connect()
        if group_id is not None:
            conn.setGroupForSession(group_id)

        trainset = conn.getObject("Dataset", trainset_id)
        trainset_path = train_path / trainset.getName()
        trainset_length = len(list(trainset.listChildren()))

        self.text_output.emit('\nDownloading data')

        centroid_list = []

        # Get images and masks
        roi_service = conn.getRoiService()
        for i, file in enumerate(trainset.listChildren()):

            QCoreApplication.processEvents()  # Update to get stop signal
            if self.stop_export:
                self.text_output.emit("Stop training set export due to user interaction.\nDelete local folder.")
                rmtree(str(trainset_path))
                break

            # Get meta data
            frame_min, frame_max, subset = 0, 65535, ''
            for ann in file.listAnnotations():
                if ann.OMERO_TYPE == omero.model.MapAnnotationI:
                    keys_values = ann.getValue()
                    for key, value in keys_values:
                        if key == 'set':
                            subset = value
                        elif key == 'min_frame':
                            frame_min = int(value)
                        elif key == 'max_frame':
                            frame_max = int(value)

            mask_rois = roi_service.findByImage(file.getId(), None)
            mask = np.zeros(shape=(file.getSizeY(), file.getSizeX()), dtype=np.uint16)
            for mask_roi in mask_rois.rois:
                for s in mask_roi.copyShapes():
                    if type(s) == omero.model.PointI:
                        x, y = s.getX().getValue(), s.getY().getValue()
                        mask[round(y), round(x)] = 1
                        centroid_list.append([file.getName(), subset, y, x])

            if np.max(mask) == 0:  # no roi found
                continue

            # Normalization
            img = file.getPrimaryPixels().getPlane(0, 0, 0)
            img = 65535 * (img.astype(np.float32) - frame_min) / (frame_max - frame_min)
            img = np.clip(img, 0, 65535).astype(np.uint16)

            # Generate overlay
            overlay = generate_overlay(img, mask)

            # Save image, mask and overlay
            fname = file.getName().split('img_')[1]
            tiff.imwrite(str(trainset_path / subset / f'img_{fname}'), img)
            tiff.imwrite(str(trainset_path / subset / f'mask_{fname}'), mask)
            tiff.imwrite(str(trainset_path / subset / f'overlay_{fname}'), overlay)

            self.progress.emit(int(100 * (i+1) / trainset_length))

        if centroid_list:
            centroid_df = pd.DataFrame(centroid_list, columns=['File', 'training set', 'y', 'x'])
            centroid_df.to_csv(str(trainset_path / 'centroids.csv'), index=False)

        if self.stop_export:
            self.progress.emit(0)
        else:
            self.progress.emit(100)

        conn.close()
        self.finished.emit()

    @pyqtSlot()
    def stop_export_process(self):
        """ Set internal export stop state to True

        :return: None
        """
        self.stop_export = True
