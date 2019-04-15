from dataset.base import BaseDataset
from dataset.utils import make_transform


class Crops(BaseDataset):
    """ Dataset of crops """

    def __init__(self, imgs, transform=None):
        """ Wrapper to a sequence of crops and labels as a torch Dataset

        Args:
            imgs: list of crops already transformed
            ys: sequence of labels
        """
        self.imgs = imgs
        # self.ys = ys
        # self.classes = set(ys)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index):
        """ Returns an element at position `index`

        Returns:
            image, label, index
        """
        # object_id = self.ys[index]
        crop = self.imgs[index]

        if self.transform:
            crop = self.transform(crop)

        return crop, -1, index

    def get_label(self, index):
        """ Gets the car ID """
        return self.ys[index]

    def nb_classes(self):
        assert len(set(self.ys)) == len(self.classes)
        return len(self.classes)
