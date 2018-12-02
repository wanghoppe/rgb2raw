from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam

config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 100
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)


## set_dir
config.TRAIN.training_dir = 'training_dir'
config.TRAIN.training_exam_dir = 'training_exam'
config.TRAIN.pretrain_checkpoint = 'checkpoint'
config.TRAIN.train_data_dir = 'dataset/Sony/Sony/short'
config.TRAIN.dark_model_dir = 'dark_model/Sony'


config.TRAIN.batch_size = 8
config.TRAIN.crop_num = 2
config.TRAIN.sample_img_size = 1000
config.TRAIN.sample_lst = list(range(0, 320, 80))

## train set location
# config.TRAIN.hr_img_path = 'data2017/DIV2K_train_HR/'
# config.TRAIN.lr_img_path = 'data2017/DIV2K_train_LR_bicubic/X4/'
#
# config.VALID = edict()
# ## test set location
# config.VALID.hr_img_path = 'data2017/DIV2K_valid_HR/'
# config.VALID.lr_img_path = 'data2017/DIV2K_valid_LR_bicubic/X4/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
