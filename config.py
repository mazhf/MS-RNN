from util.ordered_easydict import OrderedEasyDict as edict
import os
from torch.nn import Conv2d, ConvTranspose2d
import numpy as np


cfg = edict()

# ConvLSTM  TrajGRU  PredRNN  PredRNN++ MIM  MotionRNN_MIM
# Multi_Scale_Unet3plus_ConvLSTM  Multi_Scale_Fully_Connected_ConvLSTM  Muti_Scale_No_Skip_ConvLSTM
# Multi_Scale_ConvLSTM  Multi_Scale_TrajGRU  Multi_Scale_PredRNN  Multi_Scale_PredRNN++
# Multi_Scale_MIM  Multi_Scale_MotionRNN_MIM
cfg.model_name = 'ConvLSTM'
cfg.gpu = '0, 1, 2, 3'
cfg.gpu_nums = len(cfg.gpu.split(','))
cfg.work_path = 'Spatiotemporal'
cfg.dataset = 'moving-mnist-20'  # moving-mnist-20  kth_resize_png  taxiBJ  HKO-7-120-with-mask
cfg.data_path = 'Spatiotemporal'
cfg.kth_only_run = True
if 'mnist' in cfg.dataset:
    cfg.width = 64
    cfg.height = 64
elif 'kth' in cfg.dataset:
    cfg.width = 88
    cfg.height = 88
elif 'human' in cfg.dataset:
    cfg.width = 100
    cfg.height = 100
elif 'taxiBJ' in cfg.dataset:
    cfg.width = 32
    cfg.height = 32
elif 'HKO-7-120-with-mask' in cfg.dataset:
    cfg.width = 88
    cfg.height = 88
cfg.lstm_hidden_state = 66
cfg.kernel_size = 3
cfg.batch = int(4 / len(cfg.gpu.split(',')))
cfg.LSTM_conv = Conv2d
cfg.LSTM_deconv = ConvTranspose2d
cfg.CONV_conv = Conv2d
if 'mnist' in cfg.dataset:
    cfg.in_len = 10
    cfg.out_len = 10
elif 'kth' in cfg.dataset:
    cfg.in_len = 10
    cfg.out_len = 10
elif 'human' in cfg.dataset:
    cfg.in_len = 8
    cfg.out_len = 8
elif 'taxiBJ' in cfg.dataset:
    cfg.in_len = 4
    cfg.out_len = 4
elif 'HKO-7-120-with-mask' in cfg.dataset:
    cfg.in_len = 10
    cfg.out_len = 10
cfg.eval_len = 20  # only used in kth
if 'HKO' in cfg.dataset:
    cfg.epoch = 30
else:
    cfg.epoch = 20
cfg.valid_num = 10
cfg.valid_epoch = cfg.epoch // cfg.valid_num
cfg.LR = 0.0003  # 0.001
cfg.optimizer = 'Adam'
cfg.normal_mae_mse_loss = 1
cfg.dataloader_thread = 0
cfg.data_type = np.float32
cfg.use_scheduled_sampling = True
cfg.TrajGRU_link_num = 13
cfg.LSTM_layers = 6

cfg.GLOBAL = edict()
cfg.GLOBAL.MODEL_LOG_SAVE_PATH = os.path.join('/home/mazhf', cfg.work_path, 'save', cfg.dataset, cfg.model_name)
cfg.GLOBAL.DATASET_PATH = os.path.join('/home/mazhf', cfg.data_path, 'dataset', cfg.dataset)

cfg.HKO = edict()
cfg.HKO.EVALUATION = edict()
cfg.HKO.EVALUATION.THRESHOLDS = np.array([0.5, 2, 5, 10, 30])
cfg.HKO.EVALUATION.CENTRAL_REGION = (120, 120, 360, 360)
cfg.HKO.EVALUATION.BALANCING_WEIGHTS = (1, 1, 2, 5, 10, 30) 

cfg.DWD = edict()
cfg.DWD.EVALUATION = edict()
cfg.DWD.EVALUATION.THRESHOLDS = np.array([0.5, 2, 5, 10, 30])
cfg.DWD.EVALUATION.CENTRAL_REGION = (120, 120, 360, 360)
cfg.DWD.EVALUATION.BALANCING_WEIGHTS = np.array([1, 1, 2, 5, 10, 30]).astype(np.float32) 
