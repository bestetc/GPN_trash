""" Module contain tools for data loading and helpful os operation. 

"""

import os

import matplotlib.pyplot as plt
import numpy as np


def imagenette_loader(path):
    ''' Load Imagenette-320 version 2 dataset to path '''
    try:
        from fastai.vision.all import untar_data, URLs
    except ImportError:
        print('FastAI do not found')
        print('Please install Fast AI')
        print('Command for installation: "conda install -c fastai -c pytorch fastai"')
    untar_data(URLs.IMAGENETTE_320, dest=path)

def visualize_tensor(img):
    ''' Show the tensor as image. '''
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()

def create_dir(dir_path):
    ''' Create the directory '''
    try:
        os.mkdir(dir_path)
    except OSError:
        return None
    return None

def label_func(label):
    ''' Return the name of class by it's label. '''
    labels_dict = {
    0: 'tench',
    1: 'English springer',
    2: 'cassette player',
    3: 'chain saw',
    4: 'church',
    5: 'French horn',
    6: 'garbage truck',
    7: 'gas pump',
    8: 'golf ball',
    9: 'parachute'
    }
    return labels_dict[label]

def label_dirs(dirs_name):
    ''' Return the name of class by the name of dirs where the image saved. '''
    labels_dict = {
    'n01440764': 'tench',
    'n02102040': 'English springer',
    'n02979186': 'cassette player',
    'n03000684': 'chain saw',
    'n03028079': 'church',
    'n03394916': 'French horn',
    'n03417042': 'garbage truck',
    'n03425413': 'gas pump',
    'n03445777': 'golf ball',
    'n03888257': 'parachute'
    }
    return labels_dict[dirs_name]
