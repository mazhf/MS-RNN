import torch
from model import Model
from net_params import params
from util.visualization import save_movie, save_image
from config import cfg
import os
import numpy as np
from util.load_data import load_data
from torch.utils.data import DataLoader
from collections import OrderedDict


def test_demo(random_seed):
    # 单独执行，挑选random seed
    batch_size = cfg.batch
    threads = cfg.dataloader_thread
    _, _, test_data = load_data()
    test_loader = DataLoader(test_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=True)

    def normalize_data_cuda(batch):
        batch = batch.permute(1, 0, 2, 3, 4)  # S x B x C x H x W
        batch = batch / 255.0
        return batch.cuda()

    model_path = os.path.join(cfg.GLOBAL.MODEL_LOG_SAVE_PATH, 'models',
                              'epoch_' + str(cfg.valid_epoch * cfg.valid_num) + '.pth')
    print('demo_model_path:', model_path)
    test_save_path = os.path.join(cfg.GLOBAL.MODEL_LOG_SAVE_PATH, 'random_seed_' + str(random_seed) + '_demo')
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    model = Model(params[0], params[1], params[2]).cuda()
    multi_GPU_dict = torch.load(model_path)
    single_GPU_dict = OrderedDict()
    for k, v in multi_GPU_dict.items():
        single_GPU_dict[k[7:]] = v  # 去掉module.
    model.load_state_dict(single_GPU_dict, strict=True)
    i = 1
    while True:
        if i == random_seed:
            test_batch = list(test_loader)[i]
            test_batch = normalize_data_cuda(test_batch)
            input = test_batch
            break
        i += 1
    with torch.no_grad():
        output = model([input, 0])
    output = np.clip(output.cpu().numpy(), 0.0, 1.0)
    # S*B*1*H*W
    input = input[1:, 0, 0, :, :]
    input = input.cpu().numpy()
    output = output[:, 0, 0, :, :]
    in_out = []
    for i in range(input.shape[0]):
        in_out_elem = np.concatenate((input[i, ...], output[i, ...]), axis=1)
        in_out.append(in_out_elem)
    in_out = np.array(in_out)
    save_movie(data=input, save_path=os.path.join(test_save_path, 'truth.avi'))
    save_movie(data=output, save_path=os.path.join(test_save_path, 'pred.avi'))
    save_movie(data=in_out, save_path=os.path.join(test_save_path, 'truth_pred.avi'))
    save_image(data=input, save_path=os.path.join(test_save_path, 'truth_img'))
    save_image(data=output, save_path=os.path.join(test_save_path, 'pred_img'))
    save_image(data=in_out, save_path=os.path.join(test_save_path, 'truth_pred_img'))
    print('save movies and images done!')


if __name__ == '__main__':

    test_demo(cfg.test_demo_random_seed[3])
