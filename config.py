from util.ordered_easydict import OrderedEasyDict as edict
import os
from torch.nn import Conv2d, ConvTranspose2d
import numpy as np

# Preliminary!
# pip install openpyxl
# Install Local Attention: https://github.com/zzd1992/Image-Local-Attention

# ConvLSTM  TrajGRU  PredRNN  PredRNN++  MIM  MotionRNN  PredRNN-V2  PrecipLSTM  CMS-LSTM  MoDeRNN
# MS-ConvLSTM  MS-TrajGRU  MS-PredRNN  MS-PredRNN++  MS-MIM  MS-MotionRNN  MS-PredRNN-V2  MS-PrecipLSTM  MS-CMS-LSTM  MS-MoDeRNN
# MS-ConvLSTM-WO-Skip  MS-ConvLSTM-UNet3+  MS-ConvLSTM-FC
# MS-LSTM  MK-LSTM

cfg = edict()
cfg.model_name = 'ConvLSTM'
cfg.gpu = '0, 1, 2, 3'
cfg.gpu_nums = len(cfg.gpu.split(','))
cfg.work_path = 'MS-RNN'
cfg.dataset = 'DWD-12-480'  # moving-mnist-20  kth_160_png  taxiBJ  HKO-7-180-with-mask  MeteoNet-120  DWD-12-480  RAIN-F
if ('HKO' in cfg.dataset) or ('MeteoNet' in cfg.dataset) or ('DWD' in cfg.dataset) or ('RAIN-F' in cfg.dataset):
    cfg.data_path = 'Precipitation-Nowcasting'
else:
    cfg.data_path = 'Spatiotemporal'
cfg.lstm_hidden_state = 64
cfg.kernel_size = 3
cfg.batch = int(4 / len(cfg.gpu.split(',')))
cfg.LSTM_conv = Conv2d
cfg.LSTM_deconv = ConvTranspose2d
cfg.CONV_conv = Conv2d
if 'mnist' in cfg.dataset:
    cfg.width = 64
    cfg.height = 64
    cfg.in_len = 10
    cfg.out_len = 10
    cfg.epoch = 20
elif 'kth' in cfg.dataset:
    cfg.kth_only_run = True
    cfg.width = 88
    cfg.height = 88
    cfg.in_len = 10
    cfg.out_len = 10
    cfg.eval_len = 10
    cfg.epoch = 20
elif 'human' in cfg.dataset:
    cfg.width = 100
    cfg.height = 100
    cfg.in_len = 8
    cfg.out_len = 8
    cfg.epoch = 0
elif 'taxiBJ' in cfg.dataset:
    cfg.width = 32
    cfg.height = 32
    cfg.in_len = 4
    cfg.out_len = 4
    cfg.epoch = 20
elif 'HKO' in cfg.dataset:
    cfg.width = 128
    cfg.height = 128
    cfg.in_len = 10
    cfg.out_len = 10
    cfg.epoch = 30
elif 'MeteoNet' in cfg.dataset:
    cfg.width = 120
    cfg.height = 120
    cfg.in_len = 10
    cfg.out_len = 10
    cfg.epoch = 13
elif 'DWD' in cfg.dataset:
    cfg.width = 124
    cfg.height = 124
    cfg.in_len = 5
    cfg.out_len = 5
    cfg.epoch = 16
elif 'RAIN-F' in cfg.dataset:
    cfg.width = 120
    cfg.height = 120
    cfg.in_len = 2
    cfg.out_len = 2
    cfg.epoch = 8
cfg.early_stopping = False
cfg.early_stopping_patience = 3
if 'mnist' in cfg.dataset:
    cfg.valid_num = int(cfg.epoch * 0.5)
else:
    cfg.valid_num = int(cfg.epoch * 1)
cfg.valid_epoch = cfg.epoch // cfg.valid_num
cfg.LR = 0.0003
cfg.optimizer = 'Adam'
cfg.dataloader_thread = 0
cfg.data_type = np.float32
cfg.scheduled_sampling = True
if 'PredRNN-V2' in cfg.model_name:
    cfg.reverse_scheduled_sampling = True
else:
    cfg.reverse_scheduled_sampling = False
cfg.TrajGRU_link_num = 10
cfg.ce_iters = 5
cfg.decouple_loss_weight = 0.01
cfg.la_num = 30
cfg.LSTM_layers = 6
cfg.metrics_decimals = 3

cfg.root_path = '/home/mazhf'

cfg.GLOBAL = edict()
cfg.GLOBAL.MODEL_LOG_SAVE_PATH = os.path.join(cfg.root_path, cfg.work_path, 'save', cfg.dataset, cfg.model_name)
cfg.GLOBAL.DATASET_PATH = os.path.join(cfg.root_path, cfg.data_path, 'dataset', cfg.dataset)

cfg.HKO = edict()
cfg.HKO.THRESHOLDS = np.array([0.5, 2, 5, 10, 30])
cfg.HKO.CENTRAL_REGION = (120, 120, 360, 360)
cfg.HKO.BALANCING_WEIGHTS = (1, 1, 2, 5, 10, 30)
