import tensorlayer as tl
import tensorflow as tf
import rawpy
from model import SRGAN_g
from PIL import Image
import numpy as np
import glob
from PIL import Image
import numpy as np
from tensorlayer import logging
import os

def load_pretrain_model(sess, npz_file, network):
    '''
    Assign the given parameters to the TensorLayer network.
    Except the the ones with '_new'

    Parameters
    ----------
    sess : Session
        TensorFlow Session.
    npz_file : npz file
        Which contains the params
    network : :class:`Layer`
        The network to be assigned.
    '''
    if not os.path.exists(npz_file):
        logging.error("file {} doesn't exist.".format(npz_file))
        return

    ops = []
    data = np.load(npz_file)
    for idx, param in enumerate(data['params']):
        if '_new' in network.all_params[idx].name:
            pass
        else:
            ops.append(network.all_params[idx].assign(param))
    if sess is not None:
        sess.run(ops)
        logging.info("[*] Load {} SUCCESS!".format(npz_file))
    return network

def get_one_example(file, crop_size = 384, output_num = 4):
    '''
    Read a raw image
    return:
    ramdomly croped output_numx384x384x1 RAW, and output_numx96x96x3 RGB
    '''

    raw = rawpy.imread(file)
    raw_full = raw.raw_image_visible.astype(np.float32)
    rgb_full = raw.postprocess(use_camera_wb=True,
                          half_size=False,
                          no_auto_bright=True,
                          output_bps=8,
                          user_flip = 0)

#     print(rgb.shape)
#     print(raw.sizes)

    #crop the img
    H = raw.sizes.raw_height
    W = raw.sizes.raw_width

    rgbs = np.zeros([output_num, int(crop_size/4), int(crop_size/4), 3])
    raws = np.zeros([output_num, crop_size, crop_size, 1])

    for i in range(output_num):
        xx = np.random.randint(0, W - crop_size)
        yy = np.random.randint(0, H - crop_size)

        raw_matrix = rgb_full[yy:yy + crop_size, xx:xx + crop_size, :]
        raw_matrix = tl.prepro.imresize(raw_matrix, [int(crop_size/4), int(crop_size/4)])
        raw_matrix = raw_matrix /255
        rgbs[i] = raw_matrix

        raw_matrix = raw.raw_image_visible.astype(np.float32)
        raw_matrix = raw_matrix[yy:yy + crop_size, xx:xx + crop_size]
        raw_matrix = np.maximum(raw_matrix - 512, 0) / (16383 - 512)
#         print(raw_matrix.shape)
        raw_matrix = np.expand_dims(np.float32(raw_matrix), axis=2)
        raws[i] = raw_matrix

    return raws, rgbs
