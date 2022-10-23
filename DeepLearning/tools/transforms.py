import os
import sys
import pathlib
sys.path.append(pathlib.Path(__file__).parents[1].__str__())

from torchvision import transforms

from tools.image_tools import Rotate, Affine, Flip, Gauss


TRANSFORMS = [transforms.RandomRotation(180),
              transforms.RandomVerticalFlip(p=1),
              transforms.RandomHorizontalFlip(p=1),
              transforms.RandomPerspective(distortion_scale=0.1, p=1),
              transforms.RandomAffine(degrees=0, shear=10, scale=(0.75, 1.25)),
              transforms.GaussianBlur(5, (0.05, 0.5))]

TRANSFORMS_UNUSED = [Rotate(None),
                     Affine(10),
                     Flip("h", 1),
                     Flip("v", 1),
                     Gauss(5, (0.2, 1.0))]
