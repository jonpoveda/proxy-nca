import PIL

from dataset.base import BaseDataset
import numpy as np
from pathlib import Path
import torchvision
import os
from torch.utils.data.dataset import Dataset
import cv2

from dataset.utils import make_transform


class AIC19(BaseDataset):
    """ AI City 2019 dataset """

    def __init__(self, root, classes, transform):
        """ Dataset using AIC19 data

        Args:
            root: path to train folder
            classes: path to sequence/camera (e.g. 'S01/c001')
            classes: range of ints
        """
        BaseDataset.__init__(self, root, classes, transform)

        self.images_paths = Path(root).joinpath('S01', 'c001', 'images')
        imgs = list(self.images_paths.glob('*.png'))
        self.im_path = sorted(imgs, key=lambda path: path.stem)

        # Load GT
        # Format: [frame, ID, left, top, width, height, 1, -1, -1, -1]
        gtfile = Path(root).joinpath('S01', 'c001', 'gt', 'gt.txt')
        print('Reading GT file: {}'.format(gtfile))
        with gtfile.open('r') as file:
            gt = file.readlines()
            # Remove new lines
            gt = [line.split() for line in gt]
            # Separate values
            gt = [element.split(sep=',') for line in gt for element in line]

        # Convert to numpy array
        self.labels = np.array(gt, dtype=np.int16)
        self.ys = np.array(self.labels[:, 1], dtype=np.int16)

        # Filter by selected classes
        mask = np.zeros(self.ys.shape, dtype=np.bool)
        sample_classes = list(set(self.ys))
        print('Classes found: {}'.format(len(sample_classes)))
        for c in sample_classes[classes.start:classes.stop]:
            mask |= self.ys == c

        print('Labels found: {}'.format(len(self.ys)))
        self.ys = self.ys[mask]
        print('Labels selected: {}'.format(len(self.ys)))

        # self.ys = self.ys[,:]
        # MARTI
        # from dataset import parser
        # ys = parser.load_detections_txt(gtfile, "LTWH", 1)
        # print(ys)
        # frame = ys[1]
        # print(frame)
        # print(frame.get_id())
        # print(frame.get_ROIs())
        # exit(55)

    def get_image(self, frame):
        """ Get the image with name `frame` """
        # convert gray to rgb
        filename = '{:04d}.png'.format(frame)
        filepath = self.images_paths.joinpath(filename)

        # PIL
        # print('Reading image: {}'.format(filepath))
        im = PIL.Image.open(filepath)

        # OPENCV
        # im = cv2.imread(str(filepath))

        # im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        # if len(list(im.split())) == 1: im = im.convert('RGB')
        return im

    def get_label(self, index):
        """ Gets the car ID """
        return self.ys[index]

    def nb_classes(self):
        assert len(set(self.ys)) == len(self.classes)
        return len(self.classes)

    def __len__(self) -> int:
        return len(self.ys)

    def __getitem__(self, index):
        """ Returns an element at position `index`

        Returns:
            image, label, index
        """
        labels = self.labels[index]
        frame_num, object_id, left, top, width, height, = labels[0:6]
        bb = (top, left, top + height, left + width)

        # NOTE (jonatan@adsmurai.com) dev
        # im = self.get_image(15)
        im = self.get_image(frame_num)

        # PIL
        crop = im.crop([bb[1], bb[0], bb[3], bb[2]])

        # OPENCV
        # crop = im[bb[0]:bb[2], bb[1]:bb[3], :]

        # DEBUG: Show images with bb overlaid
        # im = cv2.rectangle(
        #     im,
        #     (left, top),
        #     (left + width, top + height),
        #     (0, 255, 0),
        #     3
        # )
        # im = cv2.resize(im, None, fx=.25, fy=.25)
        # cv2.imshow("frame", im)
        # cv2.waitKey(0)

        # DEBUG: Show crop
        # cv2.imshow("crop", crop)
        # cv2.waitKey(0)

        # DEBUG: Show crop (PIL)
        # crop.show()

        if self.transform is not None:
            crop = self.transform(crop)

        # DEBUG: Show transformed crop (PIL)
        # arr = np.transpose(crop.numpy().astype(np.uint8), (1, 2, 0))
        # cropp = PIL.Image.fromarray(arr)
        # cropp.show()

        # DEBUG: Show crop (OPENCV)
        # cv2.imshow("crop", crop)
        # cv2.waitKey(0)
        return crop, object_id, index


if __name__ == '__main__':
    ds = AIC19(
        root='/home/jon/repos/mcv/m6/proxy-nca/data/train/',
        classes=range(0, 50),
        transform=make_transform(**{
            "rgb_to_bgr": False,
            "rgb_to_hsv": True,
            "intensity_scale": [[0, 1], [0, 255]],
            "mean": [0, 0, 0],
            "std": [1, 1, 1],
        })
    )

    # ds = AIC19(
    #     root='/home/jon/repos/mcv/m6/proxy-nca/data/train/',
    #     classes='S01/c001',
    #     transform=None,
    # )

    # ds = AIC19(root='/home/jon/repos/mcv/m6/proxy-nca/data/train/',
    #            classes='S03/c011', transform=None)

    # for i in range(len(ds)):
    for i in range(1):
        ds[i]
