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

def get_one_example(gt_fn, train_label_dir, train_data_dir, dataset_dict,
                    data_num = 2, crop_num = 4, crop_size = 384):


    raw_label = rawpy.imread(train_label_dir + os.path.sep + gt_fn)
    rgb_label_full = raw_label.postprocess(use_camera_wb=True,
                                      half_size=False,
                                      no_auto_bright=True,
                                      output_bps=16,
                                      user_flip = 0)
    rgb_label_full = np.minimum(rgb_label_full/ 65535, 1.0)
    rgb_label_full = np.expand_dims(np.float32(rgb_label_full), axis=0)

    rgb_label_full = np.concatenate([rgb_label_full] * data_num, axis=0)



    H = rgb_label_full.shape[1]
    W = rgb_label_full.shape[2]

    rgbs_data_full_lst = []
    for in_fn in np.random.choice(dataset_dict[gt_fn], data_num, replace=False):
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)
        print(ratio)

        raw_data = rawpy.imread(train_data_dir + os.path.sep + in_fn)
        rgb_data = raw_data.postprocess(use_camera_wb=True,
                                      half_size=False,
                                      no_auto_bright=True,
                                      output_bps=16,
                                      user_flip = 0)

        # First time 10 and save to uint8 to loss some details.
        rgb_data = np.minimum(rgb_data/ 65535 * 10, 1.0)
        rgb_data = (rgb_data * 255).astype(np.uint8)
        rgb_data = tl.prepro.imresize(rgb_data, [int(H/4), int(W/4)])
        rgb_data = np.minimum(rgb_data.astype(np.float32) * ratio/10, 255)
        rgb_data = (rgb_data / 127.5) - 1

        rgb_data = np.expand_dims(rgb_data, axis=0)
        rgbs_data_full_lst.append(rgb_data)

    rgb_data_full = np.concatenate(rgbs_data_full_lst, axis=0)


    #crop the img
    gts = []
    ins = []

    for i in range(crop_num):
        xx = np.random.randint(0, W - crop_size)
        if not xx % 2 == 0:
            xx -= 1
        yy = np.random.randint(0, H - crop_size)
        if not yy % 2 == 0:
            yy -= 1

        gt_matrix = rgb_label_full[:, yy:yy + crop_size, xx:xx + crop_size, :]
        in_matrix = rgb_data_full[:,int(yy/4):int((yy + crop_size)/4), int(xx/4):int((xx + crop_size)/4), :]

        gts.append(gt_matrix)
        ins.append(in_matrix)

    gts_return = np.concatenate(gts, axis=0)
    ins_return = np.concatenate(ins, axis=0)

    return gts_return, ins_return


def get_one_example_fix_crop(gt_fn, train_label_dir, train_data_dir, dataset_dict,
                     crop_size = 384):


    raw_label = rawpy.imread(train_label_dir + os.path.sep + gt_fn)
    rgb_label_full = raw_label.postprocess(use_camera_wb=True,
                                      half_size=False,
                                      no_auto_bright=True,
                                      output_bps=16,
                                      user_flip = 0)
    rgb_label_full = np.minimum(rgb_label_full/ 65535, 1.0)
    rgb_label_full = np.expand_dims(np.float32(rgb_label_full), axis=0)

    H = rgb_label_full.shape[1]
    W = rgb_label_full.shape[2]
    xx = 1000
    yy = 1000

    in_exposure = float(dataset_dict[gt_fn][0][9:-5])
    gt_exposure = float(gt_fn[9:-5])
    ratio = min(gt_exposure / in_exposure, 300)

    raw_data = rawpy.imread(train_data_dir + os.path.sep + dataset_dict[gt_fn][0])
    rgb_data = raw_data.postprocess(use_camera_wb=True,
                                  half_size=False,
                                  no_auto_bright=True,
                                  output_bps=16,
                                  user_flip = 0)

    # First time 10 and save to uint8 to loss some details.
    rgb_data = np.minimum(rgb_data/ 65535 * 10, 1.0)
    rgb_data = (rgb_data * 255).astype(np.uint8)
    rgb_data = tl.prepro.imresize(rgb_data, [int(H/4), int(W/4)])
    rgb_data = np.minimum(rgb_data.astype(np.float32) * ratio/10, 255)
    rgb_data = (rgb_data / 127.5) - 1

    rgb_data = np.expand_dims(rgb_data, axis=0)


    gt_matrix = rgb_label_full[:, yy:yy + crop_size, xx:xx + crop_size, :]
    in_matrix = rgb_data[:,int(yy/4):int((yy + crop_size)/4), int(xx/4):int((xx + crop_size)/4), :]

    return gt_matrix, in_matrix

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
