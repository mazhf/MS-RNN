from models import *
from config import cfg
from collections import OrderedDict

b = cfg.batch
h = cfg.height
w = cfg.width

hs = cfg.lstm_hidden_state
if cfg.kernel_size == 5:
    k, s, p = 5, 1, 2
elif cfg.kernel_size == 3:
    k, s, p = 3, 1, 1
else:
    k, s, p = None, None, None

if cfg.model_name == 'ConvLSTM':
    rnn = ConvLSTM
elif cfg.model_name == 'TrajGRU':
    rnn = TrajGRU
elif cfg.model_name == 'PredRNN':
    rnn = PredRNN
elif cfg.model_name == 'PredRNN++':
    rnn = PredRNN_Plus2
elif cfg.model_name == 'MIM':
    rnn = MIM
elif cfg.model_name == 'MotionRNN':
    rnn = MotionRNN
elif cfg.model_name == 'CMS-LSTM':
    rnn = CMS_LSTM
elif cfg.model_name == 'MoDeRNN':
    rnn = MoDeRNN
elif cfg.model_name == 'PredRNN-V2':
    rnn = PredRNN_V2
elif cfg.model_name == 'PrecipLSTM':
    rnn = PrecipLSTM

elif cfg.model_name == 'MS-ConvLSTM-WO-Skip':
    rnn = MS_ConvLSTM_WO_Skip
elif cfg.model_name == 'MS-ConvLSTM':
    rnn = MS_ConvLSTM
elif cfg.model_name == 'MS-ConvLSTM-UNet3+':
    rnn = MS_ConvLSTM_UNet_Plus3
elif cfg.model_name == 'MS-ConvLSTM-FC':
    rnn = MS_ConvLSTM_FC

elif cfg.model_name == 'MS-TrajGRU':
    rnn = MS_TrajGRU
elif cfg.model_name == 'MS-PredRNN':
    rnn = MS_PredRNN
elif cfg.model_name == 'MS-PredRNN++':
    rnn = MS_PredRNN_Plus2
elif cfg.model_name == 'MS-MIM':
    rnn = MS_MIM
elif cfg.model_name == 'MS-MotionRNN':
    rnn = MS_MotionRNN
elif cfg.model_name == 'MS-CMS-LSTM':
    rnn = MS_CMS_LSTM
elif cfg.model_name == 'MS-MoDeRNN':
    rnn = MS_MoDeRNN
elif cfg.model_name == 'MS-PredRNN-V2':
    rnn = MS_PredRNN_V2
elif cfg.model_name == 'MS-PrecipLSTM':
    rnn = MS_PrecipLSTM

elif cfg.model_name == 'MS-LSTM':
    rnn = MS_LSTM
elif cfg.model_name == 'MK-LSTM':
    rnn = MK_LSTM
else:
    rnn = None

rnn = rnn(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)

if cfg.dataset in ['human3.6m', 'ucf50', 'sports10']:
    nets = [OrderedDict({'conv_embed': [3, hs, 1, 1, 0, 1]}),
            rnn,
            OrderedDict({'conv_fc': [hs, 3, 1, 1, 0, 1]})]
else:
    nets = [OrderedDict({'conv_embed': [1, hs, 1, 1, 0, 1]}),
            rnn,
            OrderedDict({'conv_fc': [hs, 1, 1, 1, 0, 1]})]
