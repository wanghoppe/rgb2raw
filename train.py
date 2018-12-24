
import tensorlayer as tl
import tensorflow as tf
import rawpy

from PIL import Image
import numpy as np
import glob
import importlib
from PIL import Image
from multiprocessing import Pool

from model import *
from utils import *
from dark_utils import *




import config as _config
importlib.reload(_config)
from config import config, log_config

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
training_dir = config.TRAIN.training_dir
pretrain_checkpoint = config.TRAIN.pretrain_checkpoint
train_data_dir = config.TRAIN.train_data_dir
train_label_dir = config.TRAIN.train_label_dir
dark_model_dir = config.TRAIN.dark_model_dir
training_exam_dir = config.TRAIN.training_exam_dir


crop_num = config.TRAIN.crop_num
data_num = config.TRAIN.data_num
sample_img_size = config.TRAIN.sample_img_size
sample_lst = config.TRAIN.sample_lst

tl.files.exists_or_mkdir(training_dir)
tl.files.exists_or_mkdir(pretrain_checkpoint)
tl.files.exists_or_mkdir(training_exam_dir)


def train():
    ###========================== DEFINE MODEL ============================###
    ## train inference
    # feed_forward generator
    rgb_96_input = tf.placeholder('float32', [None, 96, 96, 3], name='rgb_96_to_SRGAN_generator')
    net_g = SRGAN_g(rgb_96_input, is_train=True, reuse=False)
    rgb_384_output = dark_network((net_g.outputs + 1)/2, reuse = False)
    # label
    rgb_384_label = tf.placeholder('float32', [None, 384, 384, 3], name='rgb_384_label')


    # sample_generator
    rgb_96_sample = tf.placeholder('float32', [None, None, None, 3], name='rgb_96_sample_to_SRGAN_g_test')
    net_g_test = SRGAN_g(rgb_96_sample, is_train=False, reuse=True)
    rgb_384_sample = dark_network((net_g_test.outputs + 1)/2, reuse = True)

    # discriminator
    net_d, logits_real = SRGAN_d(rgb_384_label, is_train=True, reuse=False)
    _, logits_fake = SRGAN_d(rgb_384_output, is_train=True, reuse=True)

    # VGG
    rgb_224_label = tf.image.resize_images(
            rgb_384_label,
            size=[224, 224],
            method=0,
            align_corners=False)

    rgb_224_output = tf.image.resize_images(
            rgb_384_output,
            size=[224, 224],
            method=0,
            align_corners=False)

    net_vgg, vgg_target_emb = Vgg19_simple_api(rgb_224_label, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api(rgb_224_output, reuse=True)

    net_g.print_params(False)
    # net_g.print_layers()
    net_d.print_params(False)

    # ###========================= DEFINE TRAIN OPS =========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(rgb_384_output, rgb_384_label, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_loss = mse_loss + vgg_loss + g_gan_loss
    # g_loss = mse_loss + g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    ## SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)


    ###========================== RESTORE MODEL =============================###


    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))

    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz(sess=sess, name=training_dir + '/g_srgan.npz', network=net_g) is None:
        if tl.files.load_and_assign_npz(sess=sess, name=training_dir + '/g_srgan_init.npz', network=net_g) is None:
            load_pretrain_model(sess=sess, npz_file=pretrain_checkpoint + '/g_srgan.npz', network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=training_dir + '/d_srgan.npz', network=net_d)

    ## restore the dark model
    var_dict = dark_model_var_dict(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='DARK'))
    var_dict_1 = {
        'Variable': var_dict['1/Variable'],
        'Variable_1': var_dict['1/Variable_1'],
        'Variable_2': var_dict['1/Variable_2'],
        'Variable_3': var_dict['1/Variable_3']
    }

    for i in range(1,2):
        var_dict.pop('%i/Variable' % i)
        var_dict.pop('%i/Variable_1' % i)
        var_dict.pop('%i/Variable_2' % i)
        var_dict.pop('%i/Variable_3' % i)

    #load the pre-train dark model
    saver = tf.train.Saver(var_dict)
    saver_1 = tf.train.Saver(var_dict_1)

    ckpt = tf.train.get_checkpoint_state(dark_model_dir)
    if ckpt:
        print('loaded ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        saver_1.restore(sess, ckpt.model_checkpoint_path)

    # load vgg-model
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)

    ###========================== SELECT SAMPLE =============================###
    # 3 thread to read the dataset
    p = Pool(3)
    # load file list
    train_data_list = sorted(tl.files.load_file_list(train_label_dir, regx = '^0.*.ARW', printable = False))
    dataset_dict = get_dataset_dict(train_data_list, train_data_dir)
    ni = int(np.sqrt(len(sample_lst)))
    sample_file_name = [train_data_list[i] for i in sample_lst]

    ins_sample, gts_sample = get_inputs_labels(p = p,
                                            gt_lst = sample_file_name,
                                            train_label_dir = train_label_dir,
                                            train_data_dir = train_data_dir,
                                            dataset_dict = dataset_dict,
                                            crop_size = 1000)

    # save rgb x 200 ratio
    ins_sample_out = ((ins_sample+1)/2 * 255)

    ins_sample_out_filename = training_exam_dir + os.path.sep + 'ins_sample.png'
    if not os.path.exists(ins_sample_out_filename):
        tl.vis.save_images(ins_sample_out, [ni, ni], ins_sample_out_filename)


    gts_sample_out_file_name = training_exam_dir + os.path.sep + 'gts_sample.png'
    if not os.path.exists(gts_sample_out_file_name):
        tl.vis.save_images(gts_sample * 255, [ni, ni], gts_sample_out_file_name)


    ###============================= TRAINING ===============================###

    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)

    ###========================= initialize G ===============================###

    ## fixed learning rate
    for epoch in range(0, n_epoch_init + 1):
    # for epoch in range(0, 1 + 1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        # random shuffle the dataset
        np.random.shuffle(train_data_list)

        for idx in range(0, len(train_data_list), batch_size):
            step_time = time.time()

            batch_file_name = train_data_list[idx: idx+batch_size]
            ins_rgbs, gts_rgbs = get_inputs_labels(p=p,
                                                gt_lst = batch_file_name,
                                                train_label_dir = train_label_dir,
                                                train_data_dir = train_data_dir,
                                                dataset_dict = dataset_dict,
                                                data_num = data_num,
                                                crop_num = crop_num,
                                                crop_size = 384)

            ## update G
            errM, _ = sess.run([mse_loss, g_optim_init], {rgb_96_input: ins_rgbs, rgb_384_label: gts_rgbs})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 2 == 0):
            sample_out = sess.run(rgb_384_sample, {rgb_96_sample: ins_sample})
            sample_out = np.minimum(np.maximum(sample_out, 0), 1)
            sample_out = (sample_out * 255)

            train_raw_sample_out_file_name = training_exam_dir + os.path.sep + '/train_raw_sample_%d.png' % epoch

            print("[*] save images")
            tl.vis.save_images(sample_out, [ni, ni], train_raw_sample_out_file_name)

        ## save model
        if (epoch != 0) and (epoch % 2 == 0):
            tl.files.save_npz(net_g.all_params, name=training_dir + '/g_srgan_init.npz', sess=sess)


    ###========================= train GAN (SRGAN) =========================###

    for epoch in range(0, n_epoch + 1):
    # for epoch in range(0, 1 + 1):
        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        # random shuffle the dataset
        np.random.shuffle(train_data_list)

        for idx in range(0, len(train_data_list), batch_size):
            step_time = time.time()
            batch_file_name = train_data_list[idx: idx+batch_size]
            ins_rgbs, gts_rgbs = get_inputs_labels(p=p,
                                                gt_lst = batch_file_name,
                                                train_label_dir = train_label_dir,
                                                train_data_dir = train_data_dir,
                                                dataset_dict = dataset_dict,
                                                data_num = data_num,
                                                crop_num = crop_num,
                                                crop_size = 384)
            ## update D
            errD, _ = sess.run([d_loss, d_optim], {rgb_96_input: ins_rgbs, rgb_384_label: gts_rgbs})
            ## update G
            errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim],
                                                 {rgb_96_input: ins_rgbs, rgb_384_label: gts_rgbs})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                                                                                total_g_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 2 == 0):
            sample_out = sess.run(rgb_384_sample, {rgb_96_sample: ins_sample})
            sample_out = np.minimum(np.maximum(sample_out, 0), 1)
            sample_out = (sample_out * 255)

            train_raw_sample_out_file_name = training_exam_dir + os.path.sep + '/train_raw_sample_%d(GAN).png' % epoch

            print("[*] save images")
            tl.vis.save_images(sample_out, [ni, ni], train_raw_sample_out_file_name)

        ## save model
        if (epoch != 0) and (epoch % 2 == 0):
            tl.files.save_npz(net_g.all_params, name=training_dir + '/g_srgan.npz', sess=sess)
            tl.files.save_npz(net_d.all_params, name=training_dir + '/d_srgan.npz', sess=sess)
if __name__ == '__main__':
    train()
