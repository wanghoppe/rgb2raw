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
from multiprocessing import Pool
from functools import partial


def get_inputs_labels(p, file_dir, raw_file_list, crop_num, crop_size=384):
    '''
    Parameters
    ----------
    p : multiprocessing.Pool
        Pool instance for multiprocessing
    file_dir : str
        Data dir.
    raw_file_list : list
        List of name of the files need to read
    crop_num : int or None
        For every img, [crop_num] number of img are randomly cropped
        If None, use fix crop
    Return
    ----------
    inputs_rgbs: np.array
        Input array of shape [len(raw_file_list)*crop_num, 96, 96, 3]
    label_raws: np.array
        Label array of shape [len(raw_file_list)*crop_num, 384, 384, 1]

    '''
    inputs_rgbs_lst = []
    label_raws_lst = []
    # p = Pool(3)

    if not crop_num == None:
        p_func = partial(get_one_example, crop_size=crop_size, output_num=crop_num)
    else:
        p_func = partial(get_one_example_fix_crop, crop_size=crop_size)

    lst = p.map(p_func, [file_dir + os.path.sep + i for i in raw_file_list])

    # del(p)

    for a, b in lst:
        inputs_rgbs_lst.append(b)
        label_raws_lst.append(a)

    inputs_rgbs = np.concatenate(inputs_rgbs_lst, axis=0)
    label_raws = np.concatenate(label_raws_lst, axis=0)

    return inputs_rgbs, label_raws

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
        # print('yes'*1000)
    return network

def get_one_example(file, crop_size = 384, output_num = 4):
    '''
    Read a raw image, ramdomly croped
    return:
    1, raws:
        output_numx384x384x1 RAW
    2, rgbs:
        output_numx96x96x3 RGB
    '''

    raw = rawpy.imread(file)
    raw_full = raw.raw_image_visible.astype(np.float32)
    rgb_full = raw.postprocess(use_camera_wb=True,
                          half_size=False,
                          no_auto_bright=True,
                          output_bps=16,
                          user_flip = 0)

#     print(rgb.shape)
#     print(raw.sizes)

    #crop the img
    H = raw_full.shape[0]
    W = raw_full.shape[1]

    rgbs = []
    raws = []

    for i in range(output_num):
        xx = np.random.randint(0, W - crop_size)
        if not xx % 2 == 0:
            xx -= 1
        yy = np.random.randint(0, H - crop_size)
        if not yy % 2 == 0:
            yy -= 1

        rgb_matrix = rgb_full[yy:yy + crop_size, xx:xx + crop_size, :]

        # X200 and rescale to [0,1]
        rgb_matrix = np.minimum(rgb_matrix/ 65535 * 100, 1.0)

        # rescale to [0, 255] and resize to 1/4
        rgb_matrix = rgb_matrix * 255
        rgb_matrix = tl.prepro.imresize(rgb_matrix, [int(crop_size/4), int(crop_size/4)])

        # rescale to [-1, 1]
        rgb_matrix = (rgb_matrix / 127.5) - 1
        rgb_matrix = np.expand_dims(np.float32(rgb_matrix), axis=0)
        rgbs.append(rgb_matrix)

        raw_matrix = raw_full[yy:yy + crop_size, xx:xx + crop_size]
#         raw_matrix = np.maximum(raw_matrix - 512, 0) / (16383 - 512)
        raw_matrix = np.maximum(raw_matrix - 512, 0) / (16383 - 512)
        raw_matrix = np.minimum((raw_matrix * 200), 1.0)
#         print(raw_matrix.shape)
        raw_matrix = np.expand_dims(np.float32(raw_matrix), axis=2)
        raw_matrix = np.expand_dims(np.float32(raw_matrix), axis=0)
        raw_matrix = pack_raw_matrix(raw_matrix)
        raws.append(raw_matrix)

    rgbs_return = np.concatenate(rgbs, axis=0)
    raws_return = np.concatenate(raws, axis=0)

    return raws_return, rgbs_return

def get_one_example_fix_crop(file, crop_size = 384):
    '''
    Read a raw image, crop from (500, 500)
    return:
    1, raws:
        1x384x384x1 RAW
    2, rgbs:
        1x96x96x3 RGB
    '''

    raw = rawpy.imread(file)
    raw_full = raw.raw_image_visible.astype(np.float32)
    rgb_full = raw.postprocess(use_camera_wb=True,
                          half_size=False,
                          no_auto_bright=True,
                          output_bps=16,
                          user_flip = 0)

#     print(rgb.shape)
#     print(raw.sizes)

    #crop the img
    H = raw_full.shape[0]
    W = raw_full.shape[1]

    xx = 1000
    yy = 1000

    rgb_matrix = rgb_full[yy:yy + crop_size, xx:xx + crop_size, :]

    # X200 and rescale to [0,1]
    rgb_matrix = np.minimum(rgb_matrix/ 65535 * 100, 1.0)

    # rescale to [0, 255] and resize to 1/4
    rgb_matrix = rgb_matrix * 255
    rgb_matrix = tl.prepro.imresize(rgb_matrix, [int(crop_size/4), int(crop_size/4)])

    # rescale to [-1, 1]
    rgb_matrix = (rgb_matrix / 127.5) - 1

    rgb_matrix = np.expand_dims(np.float32(rgb_matrix), axis=0)


    ## For raw
    raw_matrix = raw_full[yy:yy + crop_size, xx:xx + crop_size]
#         raw_matrix = np.maximum(raw_matrix - 512, 0) / (16383 - 512)
    raw_matrix = np.maximum(raw_matrix - 512, 0) / (16383 - 512)
    raw_matrix = np.minimum((raw_matrix * 200), 1.0)
#         print(raw_matrix.shape)
    raw_matrix = np.expand_dims(np.float32(raw_matrix), axis=2)
    raw_matrix = np.expand_dims(np.float32(raw_matrix), axis=0)
    raw_matrix = pack_raw_matrix(raw_matrix)

    return raw_matrix, rgb_matrix

def pack_raw_matrix(im):
    # pack Bayer image to 4 channels
#     im = raw.raw_image_visible.astype(np.float32)

#     im = np.expand_dims(matrix, axis=)
    img_shape = im.shape
    H = img_shape[1]
    W = img_shape[2]

    out = np.concatenate((im[:, 0:H:2, 0:W:2, :],
                          im[:, 0:H:2, 1:W:2, :],
                          im[:, 1:H:2, 1:W:2, :],
                          im[:, 1:H:2, 0:W:2, :]), axis=3)
    return out
