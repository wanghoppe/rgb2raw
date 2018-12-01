
import tensorlayer as tl
import tensorflow as tf
import rawpy

import model
import utils

from PIL import Image
import numpy as np
import glob
import importlib
from PIL import Image

from model import *
from utils import *




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
dark_model_dir = config.TRAIN.dark_model_dir


crop_num = config.TRAIN.crop_num

tl.files.exists_or_mkdir(training_dir)
tl.files.exists_or_mkdir(pretrain_checkpoint)


def train():
    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [None, 96, 96, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [None, 384, 384, 1], name='t_target_image')

    net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    # net_g.print_params(False)￼

    net_g.print_params(False)
    # net_g.print_layers()
    net_d.print_params(False)
    # net_d.print_layers()

    ## test inference
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)


    # ###========================= DEFINE TRAIN OPS =========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    # vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    # g_loss = mse_loss + vgg_loss + g_gan_loss
    g_loss = mse_loss + g_gan_loss

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


    ###============================= TRAINING ===============================###

    # load file list
    train_data_list = sorted(tl.files.load_file_list(train_data_dir, regx = '^0.*.ARW', printable = False))

    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)

    ###========================= initialize G ===============================###

    ## fixed learning rate
    for epoch in range(0, n_epoch_init + 1):
    # for epoch in range(0, 1 + 1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        for idx in range(0, len(train_data_list), batch_size):
            step_time = time.time()

            batch_file_name = train_data_list[idx: idx+batch_size]
            inputs_rgbs, label_raws = get_inputs_labels(train_data_dir, batch_file_name, crop_num)

            ## update G
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: inputs_rgbs, t_target_image: label_raws})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)

        ## quick evaluation on train set
    #     if (epoch != 0) and (epoch % 10 == 0):
    #         out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})  #; print('gen sub-image:', out.shape, out.min(), out.max())
    #         print("[*] save images")
    #         tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)

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

        for idx in range(0, len(train_data_list), batch_size):
            step_time = time.time()
            batch_file_name = train_data_list[idx: idx+batch_size]
            inputs_rgbs, label_raws = get_inputs_labels(train_data_dir, batch_file_name, 2)
            ## update D
            errD, _ = sess.run([d_loss, d_optim], {t_image: inputs_rgbs, t_target_image: label_raws})
            ## update G
            errG, errM, errA, _ = sess.run([g_loss, mse_loss, g_gan_loss, g_optim],
                                                 {t_image: inputs_rgbs, t_target_image: label_raws})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f adv: %.6f)" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errA))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                                                                                total_g_loss / n_iter)
        print(log)

        ## quick evaluation on train set
#         if (epoch != 0) and (epoch % 10 == 0):
#             out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})  #; print('gen sub-image:', out.shape, out.min(), out.max())
#             print("[*] save images")
#             tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 2 == 0):
            tl.files.save_npz(net_g.all_params, name=training_dir + '/g_srgan.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_d.all_params, name=training_dir + '/d_srgan.npz'.format(tl.global_flag['mode']), sess=sess)
if __name__ == '__main__':
    train()
