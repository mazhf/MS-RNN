import torch
from config import cfg
import numpy as np
from util.evaluation import Evaluation
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import shutil
import pandas as pd
import time
from util.visualization import save_movie, save_image
from util.earlystopping import EarlyStopping
from thop import profile


IN_LEN = cfg.in_len
OUT_LEN = cfg.out_len
if 'kth' in cfg.dataset:
    EVAL_LEN = cfg.eval_len
gpu_nums = cfg.gpu_nums
decimals = cfg.metrics_decimals


def normalize_data_cuda(batch):
    batch = batch.permute(1, 0, 2, 3, 4)  # S x B x C x H x W
    batch = batch / 255.0
    return batch.cuda()


def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= gpu_nums
    return rt


# is main process ?
def is_master_proc(gpu_nums=gpu_nums):
    return torch.distributed.get_rank() % gpu_nums == 0


def train_and_test(model, optimizer, criterion, train_epoch, valid_epoch, loader, train_sampler):
    train_valid_metrics_save_path, model_save_path, writer, save_path, test_metrics_save_path = [None] * 5
    train_loader, test_loader, valid_loader = loader
    start = time.time()
    if 'kth' in cfg.dataset:
        eval_ = Evaluation(seq_len=IN_LEN + EVAL_LEN - 1, use_central=False)
    else:
        eval_ = Evaluation(seq_len=IN_LEN + OUT_LEN - 1, use_central=False)
    if is_master_proc():
        save_path = cfg.GLOBAL.MODEL_LOG_SAVE_PATH
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        model_save_path = os.path.join(save_path, 'models')
        os.makedirs(model_save_path)
        log_save_path = os.path.join(save_path, 'logs')
        os.makedirs(log_save_path)
        test_metrics_save_path = os.path.join(save_path, "test_metrics.xlsx")
        writer = SummaryWriter(log_save_path)
    train_loss = 0.0
    params_lis = []
    eta = 1.0
    delta = 1 / (train_epoch * len(train_loader))
    early_stopping = EarlyStopping(patience=cfg.early_stopping_patience, verbose=True)
    for epoch in range(1, train_epoch + 1):
        if is_master_proc():
            print('epoch: ', epoch)
        pbar = tqdm(total=len(train_loader), desc="train_batch", disable=not is_master_proc())
        # train
        train_sampler.set_epoch(epoch)
        model.train()
        for idx, train_batch in enumerate(train_loader, 1):
            train_batch = normalize_data_cuda(train_batch)
            optimizer.zero_grad()
            train_pred, decouple_loss = model([train_batch, eta, epoch], mode='train')
            loss = criterion(train_batch[1:, ...], train_pred, decouple_loss)
            loss.backward()
            optimizer.step()
            loss = reduce_tensor(loss)  # all reduce
            train_loss += loss.item()
            eta -= delta
            eta = max(eta, 0)
            pbar.update(1)

            # compute Params and FLOPs for Generator
            if epoch == 1 and idx == 1 and is_master_proc():
                Total_params = 0
                Trainable_params = 0
                NonTrainable_params = 0
                for param in model.parameters():
                    mulValue = param.numel()
                    Total_params += mulValue
                    if param.requires_grad:
                        Trainable_params += mulValue
                    else:
                        NonTrainable_params += mulValue
                Total_params = np.around(Total_params / 1e+6, decimals=decimals)
                Trainable_params = np.around(Trainable_params / 1e+6, decimals=decimals)
                NonTrainable_params = np.around(NonTrainable_params / 1e+6, decimals=decimals)
                # 使用nn.BatchNorm2d时，flop计算会报错，需要注释掉
                flops, _ = profile(model.module, inputs=([train_batch, eta, epoch], 'train',))
                flops = np.around(flops / 1e+9, decimals=decimals)
                params_lis.append(Total_params)
                params_lis.append(Trainable_params)
                params_lis.append(NonTrainable_params)
                params_lis.append(flops)
                print(f'Total params: {Total_params}M')
                print(f'Trained params: {Trainable_params}M')
                print(f'Untrained params: {NonTrainable_params}M')
                print(f'FLOPs: {flops}G')
        pbar.close()

        # valid
        if epoch % valid_epoch == 0:
            train_loss = train_loss / (len(train_loader) * valid_epoch)
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for valid_batch in valid_loader:
                    valid_batch = normalize_data_cuda(valid_batch)
                    valid_pred, decouple_loss = model([valid_batch, 0, train_epoch], mode='test')
                    loss = criterion(valid_batch[1:, ...], valid_pred, decouple_loss)
                    loss = reduce_tensor(loss)  # all reduce
                    valid_loss += loss.item()
            valid_loss = valid_loss / len(valid_loader)
            if is_master_proc():
                writer.add_scalars("loss", {"train": train_loss, "valid": valid_loss}, epoch)  # plot loss
                torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
            train_loss = 0.0
            # early stopping
            if cfg.early_stopping:
                early_stopping(valid_loss, model, is_master_proc())
                if early_stopping.early_stop:
                    print("Early Stopping!")
                    break

    # test
    eval_.clear_all()
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = normalize_data_cuda(test_batch)
            test_pred, decouple_loss = model([test_batch, 0, train_epoch], mode='test')
            loss = criterion(test_batch[1:, ...], test_pred, decouple_loss)
            test_loss += loss.item()
            test_batch_numpy = test_batch.cpu().numpy()
            test_pred_numpy = np.clip(test_pred.cpu().numpy(), 0.0, 1.0)
            eval_.update(test_batch_numpy[1:, ...], test_pred_numpy)

    if is_master_proc():
        test_metrics_lis = eval_.get_metrics()
        test_loss = test_loss / len(test_loader)
        test_metrics_lis.append(test_loss)
        end = time.time()
        running_time = np.around((end - start) / 3600, decimals=decimals)
        print("===============================")
        print('Running time: {} hours'.format(running_time))
        print("===============================")
        save_test_metrics(test_metrics_lis, test_metrics_save_path, params_lis, running_time)
        eval_.clear_all()

    if is_master_proc():
        writer.close()
        test_demo(test_loader, model)


def nan_to_num(metrics):
    for i in range(len(metrics)):
        metrics[i] = np.nan_to_num(metrics[i])
    return metrics


def save_test_metrics(m_lis, path, p_lis, run_tim):
    m_lis = nan_to_num(m_lis)
    col0 = ['ssim↑', 'psnr↑', 'gdl↓', 'b_mse↓', 'b_mae↓', 'mse↓', 'mae↓',
            'pod_0.5↑', 'pod_2↑', 'pod_5↑', 'pod_10↑', 'pod_30↑',
            'far_0.5↓', 'far_2↓', 'far_5↓', 'far_10↓', 'far_30↓',
            'csi_0.5↑', 'csi_2↑', 'csi_5↑', 'csi_10↑', 'csi_30↑',
            'hss_0.5↑', 'hss_2↑', 'hss_5↑', 'hss_10↑', 'hss_30↑',
            'loss↓', 'params(M)', 'trained_params(M)', 'untrained_params(M)', 'flops(G)', 'time(H)']
    if 'kth' in cfg.dataset:
        frame_wise_col0 = [str(i) for i in range(1, IN_LEN + EVAL_LEN)]
    else:
        frame_wise_col0 = [str(i) for i in range(1, IN_LEN + OUT_LEN)]
    col1 = []
    frame_wise_csi_hss_col12 = []
    mse_col1 = None
    ssim_col1 = None
    psnr_col1 = None
    for i in range(len(m_lis)):
        metric = m_lis[i]
        # float32 is not compatible with float64 of pandas and python, causing the round to fail.
        metric = metric.astype(np.float64)
        if i in [7, 8, 9, 10]:
            for j in range(len(cfg.HKO.THRESHOLDS)):
                col1.append(np.around(metric[:, j].mean(), decimals=decimals))  # pod far csi hss
                if (i in [9, 10]) and (j == len(cfg.HKO.THRESHOLDS) - 1):
                    frame_wise_csi_hss_col12.append(
                        np.around(metric[:, j], decimals=decimals))  # frame_wise csi30 hss30
        elif i == 11:
            col1.append(np.around(metric, decimals=decimals))  # loss
        else:
            col1.append(
                np.around(metric.mean(), decimals=decimals))  # ssim psnr gdl bmse bmae mse mae params flops time
            if i == 0:
                ssim_col1 = np.around(metric, decimals=decimals)  # frame_wise ssim
            elif i == 1:
                psnr_col1 = np.around(metric, decimals=decimals)  # frame_wise psnr
            elif i == 5:
                mse_col1 = np.around(metric, decimals=decimals)  # frame_wise mse

    # all
    col1 += p_lis
    col1.append(run_tim)
    df = pd.DataFrame()
    df['0'] = col0
    df['1'] = col1
    df.columns = ['Metrics', 'Value']
    df.to_excel(path, index=False)

    # frame-wise csi30 hss30
    csi_hss_df = pd.DataFrame()
    csi_hss_df['0'] = frame_wise_col0
    csi_hss_df['1'] = frame_wise_csi_hss_col12[0]
    csi_hss_df['2'] = frame_wise_csi_hss_col12[1]
    csi_hss_df.columns = ['frame', 'csi↑', 'hss↑']
    split = path.split('.')
    csi_hss_path = split[0] + '_framewise_csi30_hss30.' + split[1]
    csi_hss_df.to_excel(csi_hss_path, index=False)

    # frame-wise mse
    mse_df = pd.DataFrame()
    mse_df['0'] = frame_wise_col0
    mse_df['1'] = mse_col1
    mse_df.columns = ['frame', 'mse↓']
    mse_path = split[0] + '_framewise_mse.' + split[1]
    mse_df.to_excel(mse_path, index=False)

    # frame-wise ssim
    ssim_df = pd.DataFrame()
    ssim_df['0'] = frame_wise_col0
    ssim_df['1'] = ssim_col1
    ssim_df.columns = ['frame', 'ssim↑']
    ssim_path = split[0] + '_framewise_ssim.' + split[1]
    ssim_df.to_excel(ssim_path, index=False)

    # frame-wise psnr
    psnr_df = pd.DataFrame()
    psnr_df['0'] = frame_wise_col0
    psnr_df['1'] = psnr_col1
    psnr_df.columns = ['frame', 'psnr↑']
    psnr_path = split[0] + '_framewise_psnr.' + split[1]
    psnr_df.to_excel(psnr_path, index=False)


def test_demo(test_loader, model):
    model.eval()
    with torch.no_grad():
        for i in range(len(test_loader)):
            test_batch = list(test_loader)[i]
            test_batch = normalize_data_cuda(test_batch)
            input = test_batch
            output, _ = model([input, 0, cfg.epoch], mode='test')
            input = input[:, 0, ...].cpu().numpy()  # s c h w. When batch=2, only half is saved!
            output = np.clip(output[:, 0, ...].cpu().numpy(), 0.0, 1.0)  # s-1 c h w
            output = np.concatenate([input[0, ...][np.newaxis, ...], output], axis=0)  # s c h w
            in_out = []
            for j in range(input.shape[0]):
                in_out_frm = np.concatenate((input[j, ...], output[j, ...]), axis=2)  # s c h w
                in_out.append(in_out_frm)
            in_out = np.array(in_out)  # s c h w
            test_demo_save_path = os.path.join(cfg.GLOBAL.MODEL_LOG_SAVE_PATH, 'demo',
                                               'random_seed_' + str(i + 1) + '_demo')
            if not os.path.exists(test_demo_save_path):
                os.makedirs(test_demo_save_path)
            save_movie(data=input, save_path=os.path.join(test_demo_save_path, 'truth.avi'))
            save_movie(data=output, save_path=os.path.join(test_demo_save_path, 'pred.avi'))
            save_movie(data=in_out, save_path=os.path.join(test_demo_save_path, 'truth_pred.avi'))
            save_image(data=input, save_path=os.path.join(test_demo_save_path, 'truth_img'))
            save_image(data=output, save_path=os.path.join(test_demo_save_path, 'pred_img'))
            save_image(data=in_out, save_path=os.path.join(test_demo_save_path, 'truth_pred_img'))
    print('save movies and images done!')
