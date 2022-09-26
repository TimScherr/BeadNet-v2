import numpy as np

from copy import deepcopy
from omero.gateway import BlitzGateway
from PyQt5.QtCore import pyqtSignal, QObject, pyqtSlot
from random import randint
from skimage.transform import rescale


class DataCropWorker(QObject):
    """ Worker class for training crop creation """
    finished = pyqtSignal()  # Signal when cropping is finished
    text_output = pyqtSignal(str)  # Signal for possible exceptions, e.g., too small crops
    crops = pyqtSignal(object)  # Signal for sending the crops
    stop_creation = False
    
    def __init__(self, img_list, crop_size, scale_factor, trainset_id, train_path, omero_username, omero_password, omero_host,
                 omero_port, group_id):
        """

        :param img_list: List with information about the images to crop
        :type img_list: list
        :param crop_size: Crop size to use for cropping
        :type crop_size: int
        :param scale_factor: Scale factor for upsampling
        :type scale_factor: int
        :param trainset_id: OMERO id of the training dataset to use
        :type trainset_id: int
        :param train_path: Local path for training
        :type train_path: pathlib Path object
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
        
        self.img_list = img_list
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.trainset_id = trainset_id
        self.train_path = train_path
        self.img_idx = 0
        self.crop = None
        self.omero_username = omero_username
        self.omero_password = omero_password
        self.omero_host = omero_host
        self.omero_port = omero_port
        self.group_id = group_id
        self.conn = None

        # Connect to omero
        self.connect()

        # Load first crop --> blocks main thread if called during initialization
        # self.next_crop()

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

    def get_crop(self):
        """ Emit crop(s) """

        # Emit loaded crop
        self.crops.emit(deepcopy(self.crop))

        # Get next crop
        if self.img_idx == len(self.img_list):
            self.text_output.emit('No more crops can be found. Start process again to see not accepted crops.')
            self.crop_creation_finished()
        else:
            self.next_crop()

    def next_crop(self):
        """ Generate next crop(s) """

        # Try to get crops
        cropping = True
        while cropping and self.img_idx < len(self.img_list):

            # Load image from omero
            try:
                img = self.conn.getObject("Image", self.img_list[self.img_idx]['id'])
            except Exception as e:  # probably timeout  --> reconnect and try again
                self.disconnect()
                self.connect()
                img = self.conn.getObject("Image", self.img_list[self.img_idx]['id'])

            # Maximum intensity projection for z-stacks
            slice_list = []
            if img.getSizeZ() > 1:
                for slice_idx in range(img.getSizeZ()):
                    slice_list.append(img.getPrimaryPixels().getPlane(slice_idx,
                                                                      self.img_list[self.img_idx]['channel'],
                                                                      self.img_list[self.img_idx]['frame']))
                img = np.array(slice_list)
                img = np.max(img, axis=0)
            else:
                img = img.getPrimaryPixels().getPlane(0,
                                                      self.img_list[self.img_idx]['channel'],
                                                      self.img_list[self.img_idx]['frame'])

            if self.scale_factor > 1:
                img = rescale(img, self.scale_factor, preserve_range=True, order=2).astype(img.dtype)

            # Check which dimension is larger --> crop_dimension
            if img.shape[0] > img.shape[1]:
                crop_dim = 0
            else:
                crop_dim = 1

            # Check if 3 crops are possible
            if img.shape[crop_dim] > 3 * self.crop_size:
                n_crops = 3
            elif img.shape[crop_dim] > 2 * self.crop_size:
                n_crops = 2
            else:
                n_crops = 1

            # Get frame_min and frame_max before padding/cropping
            img_min, img_max, img_mean, img_std = np.min(img), np.max(img), np.mean(img), np.std(img)

            # Check if padding is needed
            if 0.9 * self.crop_size > img.shape[0] or 0.9 * self.crop_size > img.shape[1]:  # Skip too small images
                self.img_idx += 1
                continue
            x_pads = np.maximum(0, self.crop_size - img.shape[1])
            y_pads = np.maximum(0, self.crop_size - img.shape[0])
            img = np.pad(img, ((0, y_pads), (0, x_pads)), mode='constant', constant_values=img_min)

            # Get crop starting points
            crops = []
            for i in range(n_crops):

                c = img.shape[crop_dim] // n_crops

                if x_pads > 0 and x_pads > 0:
                    a, b = 0, 0
                elif crop_dim == 0 and y_pads == 0 and img.shape[crop_dim] > self.crop_size:
                    a = randint(i * c,
                                np.minimum(img.shape[crop_dim] - self.crop_size, (i + 1) * c - self.crop_size))
                    b = randint(0, img.shape[1] - self.crop_size)
                elif crop_dim == 1 and x_pads == 0 and img.shape[crop_dim] > self.crop_size:
                    a = randint(0, img.shape[0] - self.crop_size)
                    b = randint(i * c,
                                np.minimum(img.shape[crop_dim] - self.crop_size, (i + 1) * c - self.crop_size))
                else:
                    a, b = 0, 0

                # Crop
                crop = img[a:a + self.crop_size, b:b + self.crop_size]

                # In gui shown crop
                crop_show = 255 * (crop.astype(np.float32) - img_min) / (img_max - img_min)
                crop_show = crop_show.astype(np.uint8)

                roi, roi_show = None, None

                # Add info and crops as dict to list
                crops.append({'project': str(self.img_list[self.img_idx]['project']),
                              'dataset': str(self.img_list[self.img_idx]['dataset']),
                              'image': str(self.img_list[self.img_idx]['name']),
                              'image_id': self.img_list[self.img_idx]['id'],  # convert to string later --> 'used' key
                              'crop_size': str(self.crop_size),
                              'channel': self.img_list[self.img_idx]['channel'],   # convert to string later
                              'frame': self.img_list[self.img_idx]['frame'],   # convert to string later --> 'used' key
                              'max_frame': str(img_max),
                              'mean_frame': str(img_mean),
                              'min_frame': str(img_min),
                              'std_frame': str(img_std),
                              'mip': str(len(slice_list) > 0),
                              'scale_factor': str(self.scale_factor),
                              'x_start': str(b),
                              'y_start': str(a),
                              'img': img[a:a + self.crop_size, b:b + self.crop_size],
                              'img_show': crop_show,
                              'roi': roi,
                              'roi_show': roi_show})

            self.img_idx += 1
            cropping = False
            self.crop = crops

    @pyqtSlot()
    def crop_creation_finished(self):
        """ Close connection and send finished signal """
        self.disconnect()
        self.finished.emit()

    @pyqtSlot()
    def stop_crop_process(self):
        """ Set internal stop state to True

        :return: None
        """
        self.stop_creation = True
