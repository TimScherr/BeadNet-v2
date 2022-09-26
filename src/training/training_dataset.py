import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    """ Pytorch data set for instance cell nuclei segmentation """

    def __init__(self, root_dir, label_type, mode='train', transform=lambda x: x):
        """

        :param root_dir: Directory containing all created training/validation data sets.
            :type root_dir: pathlib Path object.
        :param mode: 'train' or 'val'.
            :type mode: str
        :param transform: transforms.
            :type transform:
        :return: Dict (image, label, id).
        """

        self.img_ids = sorted((root_dir / mode).glob('img*.tif'))
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transform
        self.label_type = label_type

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        # Get image id
        img_id = self.img_ids[idx]

        # Load image
        img = tiff.imread(str(img_id))

        # Load label image
        label_id = img_id.parent / f"{self.label_type}{img_id.name.split('img')[-1]}"
        label_img = tiff.imread(str(label_id)).astype(np.uint8)

        # Channel dimension needed later (for pytorch)
        img = img[..., None]
        label_img = label_img[..., None]

        sample = {'image': img,
                  'label': label_img,
                  'id': img_id.stem}

        sample = self.transform(sample)

        return sample
