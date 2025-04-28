import numpy as np
from scipy import io
import utils.DM as dm

def load_data(path, type_dm, path1):
    if type_dm:
        image, dimensions, calibration, metadata = dm.dm_load(path)
        image = image.transpose([1, 2, 0])
    else:
        image1 = io.loadmat(path)
        image = np.array(image1[path1], dtype='float32')
    return image