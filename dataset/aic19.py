from dataset.base import BaseDataset
import numpy as np
from pathlib import Path
import torchvision
import os


class AIC19(BaseDataset):
    """ AI City 2019 dataset """

    def __init__(self, root, classes, transform=None):
        BaseDataset.__init__(self, root, classes, transform)

        images_paths = Path(root).joinpath('images')
        imgs = list(images_paths.glob('*.png'))
        imgs = sorted(imgs, key=lambda path: path.stem)

        image_names = [p.stem for p in imgs]

        # Load GT
        # Format: [frame, ID, left, top, width, height, 1, -1, -1, -1]
        gtfile = Path(root).joinpath('gt', 'gt.txt')
        with gtfile.open('r') as file:
            gt = file.readlines()
            # Remove new lines
            gt = [line.split() for line in gt]
            # Separate values
            gt = [element.split(sep=',') for line in gt for element in line]
            # Convert to numpy array
            gt = np.array(gt, dtype=np.int16)



if __name__ == '__main__':
    AIC19(root='/home/jon/repos/mcv/m6/proxy-nca/data/train/S01/c001',
          classes=range(1, 4), transform=None)
