from .cars import Cars
from .cub import CUBirds
from .aic19 import AIC19
from . import utils
from .base import BaseDataset

_type = {
    'cars': Cars,
    'cub': CUBirds,
    'aic19': AIC19,
}


def load(name, root, classes, transform=None):
    return _type[name](root=root, classes=classes, transform=transform)
