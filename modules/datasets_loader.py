import os
import matplotlib.pyplot as plt
import numpy as np

def imagenette_loader(path, need_install=False):
    if need_install == True:
        os.system('conda install -c fastai -c pytorch fastai')
    
    dataset_path = untar_data(URLs.IMAGENETTE_320, dest=path)
    return dataset_path

# labels_dict = dict(
#     n01440764='tench',
#     n02102040='English springer',
#     n02979186='cassette player',
#     n03000684='chain saw',
#     n03028079='church',
#     n03394916='French horn',
#     n03417042='garbage truck',
#     n03425413='gas pump',
#     n03445777='golf ball',
#     n03888257='parachute'
# )

labels_dict = {
    0:'tench',
    1:'English springer',
    2:'cassette player',
    3:'chain saw',
    4:'church',
    5:'French horn',
    6:'garbage truck',
    7:'gas pump',
    8:'golf ball',
    9:'parachute'
}

def label_func(label):
    return labels_dict[label]

def visual_tensor(img):
    img = img / 2 + 0.5     
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def create_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        print('already created')
    else:
        print(dir_path + 'created')