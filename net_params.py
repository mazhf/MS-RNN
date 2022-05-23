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

if cfg.model_name == 'TrajGRU':
    rnn_param = TrajGRU(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'MIM':
    rnn_param = MIM(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'PredRNN':
    rnn_param = PredRNN(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'PredRNN++':
    rnn_param = PredRNN_plus2(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'ConvLSTM':
    rnn_param = ConvLSTM(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'MotionRNN_MIM':
    rnn_param = MotionRNN_MIM(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Scale_ConvLSTM':
    rnn_param = Multi_Scale_ConvLSTM(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Scale_Unet3plus_ConvLSTM':
    rnn_param = Multi_Scale_Unet3plus_ConvLSTM(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Scale_Fully_Connected_ConvLSTM':
    rnn_param = Multi_Scale_Fully_Connected_ConvLSTM(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Muti_Scale_No_Skip_ConvLSTM':
    rnn_param = Muti_Scale_No_Skip_ConvLSTM(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Scale_PredRNN':
    rnn_param = Multi_Scale_PredRNN(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Scale_MIM':
    rnn_param = Multi_Scale_MIM(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Scale_PredRNN++':
    rnn_param = Multi_Scale_PredRNN_plus2(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Scale_MotionRNN_MIM':
    rnn_param = Multi_Scale_MotionRNN_MIM(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)
elif cfg.model_name == 'Multi_Scale_TrajGRU':
    rnn_param = Multi_Scale_TrajGRU(input_channel=hs, output_channel=hs, b_h_w=(b, h, w), kernel_size=k, stride=s, padding=p)



if cfg.dataset in ['human3.6m', 'ucf50', 'sports10']:
    params = [OrderedDict({'conv_embed': [3, hs, 1, 1, 0, 1]}),
              rnn_param,
              OrderedDict({'conv_fc': [hs, 3, 1, 1, 0, 1]})]
else:
    params = [OrderedDict({'conv_embed': [1, hs, 1, 1, 0, 1]}),
              rnn_param,
              OrderedDict({'conv_fc': [hs, 1, 1, 1, 0, 1]})]
