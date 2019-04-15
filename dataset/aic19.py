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
        """
        BaseDataset.__init__(self, root, classes, transform)

        self.images_paths = Path(root).joinpath(classes, 'images')
        imgs = list(self.images_paths.glob('*.png'))
        self.im_path = sorted(imgs, key=lambda path: path.stem)
        # image_names = [p.stem for p in imgs]

        # Load GT
        # Format: [frame, ID, left, top, width, height, 1, -1, -1, -1]
        gtfile = Path(root).joinpath(classes, 'gt', 'gt.txt')
        print('Reading GT file: {}'.format(gtfile))
        with gtfile.open('r') as file:
            gt = file.readlines()
            # Remove new lines
            gt = [line.split() for line in gt]
            # Separate values
            gt = [element.split(sep=',') for line in gt for element in line]

        # Convert to numpy array
        self.labels = np.array(gt, dtype=np.int16)
        self.ys = self.labels[:, 1]

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
        print('Reading image: {}'.format(filepath))

        # PIL
        im = PIL.Image.open(filepath)

        # OPENCV
        # im = cv2.imread(str(filepath))

        # im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        # if len(list(im.split())) == 1: im = im.convert('RGB')
        return im

    def get_label(self, index):
        """ Gets the car ID """
        return self.ys[index]

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
        # im = self.get_image(frame_num)
        im = self.get_image(15)

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
        classes='S01/c001',
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
