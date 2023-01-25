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
import cv2


IN_LEN = cfg.in_len
OUT_LEN = cfg.out_len
EVAL_LEN = cfg.eval_len
gpu_nums = cfg.gpu_nums


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


def train_and_test(model, optimizer, criterion, train_epoch, valid_epoch, save_checkpoint_epoch, loader, train_sampler):
    """
    只在主进程创建，删除，写入文件。loss和度量是all_reduce后的结果，同步到每个进程，这样主进程中的就是均值
    """
    train_valid_metrics_save_path, model_save_path, writer, save_path, test_metrics_save_path = [None] * 5
    train_loader, test_loader, valid_loader = loader
    start = time.time()
    # 初始化metrics, 加载数据类, 此处直接使用测试集作为验证集，减少工作量，但会增加训练时间，训练完毕，即可得到测试集上的度量！！！
    if 'kth' in cfg.dataset:
        eval_ = Evaluation(seq_len=IN_LEN + EVAL_LEN - 1, use_central=False)
    else:
        eval_ = Evaluation(seq_len=IN_LEN + OUT_LEN - 1, use_central=False)
    if is_master_proc():
        # 初始化保存路径，覆盖前面训练的models, logs, metrics
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
    # 训练、验证
    train_loss = 0.0
    valid_times = 0
    params_lis = []
    eta = 1.0
    delta = 1 / (train_epoch * len(train_loader))

    for epoch in range(1, train_epoch + 1):
        if is_master_proc():
            print('epoch: ', epoch)
        pbar = tqdm(total=len(train_loader), desc="train_batch", disable=not is_master_proc())  # 进度条
        # train
        train_sampler.set_epoch(epoch)  # 防止每个epoch被分配到每块卡上的数据都一样，虽然数据已经平分给各个卡，但不设置的话，每次平分的数据都一样
        for idx, train_batch in enumerate(train_loader, 1):
            train_batch = normalize_data_cuda(train_batch)
            model.train()
            optimizer.zero_grad()
            train_pred = model([train_batch, eta], mode='train')
            loss = criterion(train_batch[1:, ...], train_pred, epoch)
            loss.backward()
            optimizer.step()
            # 更新lr、loss、metrics
            loss = reduce_tensor(loss)  # all reduce
            train_loss += loss.item()
            eta -= delta
            eta = max(eta, 0)
            pbar.update(1)

            # 计算参数量
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
                params_lis.append(Total_params)
                params_lis.append(Trainable_params)
                params_lis.append(NonTrainable_params)
                print(f'Total params: {Total_params}')
                print(f'Trainable params: {Trainable_params}')
                print(f'Non-trainable params: {NonTrainable_params}')
        pbar.close()

        # valid
        if epoch % valid_epoch == 0:
            valid_times += 1
            train_loss = train_loss / (len(train_loader) * valid_epoch)
            with torch.no_grad():
                model.eval()
                valid_loss = 0.0
                for valid_batch in valid_loader:
                    valid_batch = normalize_data_cuda(valid_batch)
                    valid_pred = model([valid_batch, 0], mode='test')
                    loss = criterion(valid_batch[1:, ...], valid_pred, None)
                    loss = reduce_tensor(loss)  # all reduce
                    valid_loss += loss.item()
                valid_loss = valid_loss / len(valid_loader)
            # 第一个参数可以简单理解为保存tensorboard中图的名称，第二个参数是可以理解为Y轴数据，第三个参数可以理解为X轴数据。
            if is_master_proc():
                writer.add_scalars("loss", {"train": train_loss, "valid": valid_loss}, epoch)  # plot loss
            train_loss = 0.0

        # save model
        if is_master_proc() and epoch % save_checkpoint_epoch == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))

        # test
        if epoch == train_epoch:
            eval_.clear_all()
            with torch.no_grad():
                model.eval()
                test_loss = 0.0
                test_times = 0
                for test_batch in test_loader:
                    test_times += 1
                    test_batch = normalize_data_cuda(test_batch)
                    test_pred = model([test_batch, 0], mode='test')
                    loss = criterion(test_batch[1:, ...], test_pred, None)
                    test_loss += loss.item()
                    test_batch_numpy = test_batch.cpu().numpy()
                    test_pred_numpy = np.clip(test_pred.detach().cpu().numpy(), 0.0, 1.0)
                    eval_.update(test_batch_numpy[1:, ...], test_pred_numpy)

                if is_master_proc():
                    test_metrics_lis = eval_.get_metrics()
                    test_loss = test_loss / test_times
                    test_metrics_lis.append(test_loss)
                    end = time.time()
                    running_time = round((end - start) / 3600, 2)
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
    col0 = ['test_ssim', 'test_psnr', 'test_gdl', 'test_balanced_mse', 'test_balanced_mae', 'test_mse', 'test_mae',
            'test_pod_0.5', 'test_pod_2', 'test_pod_5', 'test_pod_10', 'test_pod_30',
            'test_far_0.5', 'test_far_2', 'test_far_5', 'test_far_10', 'test_far_30',
            'test_csi_0.5', 'test_csi_2', 'test_csi_5', 'test_csi_10', 'test_csi_30',
            'test_hss_0.5', 'test_hss_2', 'test_hss_5', 'test_hss_10', 'test_hss_30',
            'test_loss', 'Total_params', 'Trainable_params', 'NonTrainable_params', 'time']
    if 'kth' in cfg.dataset:
        add_col0 = [str(i) for i in range(1, IN_LEN + EVAL_LEN)]
    else:
        add_col0 = [str(i) for i in range(1, IN_LEN + OUT_LEN)]
    col1 = []
    add_col1 = []
    for i in range(len(m_lis)):
        metric = m_lis[i]
        if i in [7, 8, 9, 10]:
            for j in range(len(cfg.HKO.EVALUATION.THRESHOLDS)):
                col1.append(metric[:, j].mean())  # pod far csi hss
                if (i in [9, 10]) and (j == len(cfg.HKO.EVALUATION.THRESHOLDS) - 1):
                    add_col1.append(metric[:, j])
        elif i == 11:
            col1.append(metric)  # loss
        else:
            col1.append(metric.mean())  # ssim psnr gdl bmse bmae mse mae
            if i == 0:
                ssim_col1 = metric
            elif i == 1:
                psnr_col1 = metric
            elif i == 5:
                add_add_col1 = metric

    # all
    col1 += p_lis
    col1.append(run_tim)
    df = pd.DataFrame()
    df['0'] = col0
    df['1'] = col1
    df.columns = ['Metrics', 'Value']
    df.to_excel(path, index=0)

    # frame-wise csi30 hss30
    add_df = pd.DataFrame()
    add_df['0'] = add_col0
    add_df['1'] = add_col1[0]
    add_df['2'] = add_col1[1]
    add_df.columns = ['frame', 'csi', 'hss']
    split = path.split('.')
    add_path = split[0] + '_framewise_csi30_hss30.' + split[1]
    add_df.to_excel(add_path, index=0)

    # frame-wise mse
    add_add_df = pd.DataFrame()
    add_add_df['0'] = add_col0
    add_add_df['1'] = add_add_col1
    add_add_df.columns = ['frame', 'mse']
    add_add_path = split[0] + '_framewise_mse.' + split[1]
    add_add_df.to_excel(add_add_path, index=0)

    # frame-wise ssim
    ssim_df = pd.DataFrame()
    ssim_df['0'] = add_col0
    ssim_df['1'] = ssim_col1
    ssim_df.columns = ['frame', 'ssim']
    ssim_path = split[0] + '_framewise_ssim.' + split[1]
    ssim_df.to_excel(ssim_path, index=0)

    # frame-wise psnr
    psnr_df = pd.DataFrame()
    psnr_df['0'] = add_col0
    psnr_df['1'] = psnr_col1
    psnr_df.columns = ['frame', 'psnr']
    psnr_path = split[0] + '_framewise_psnr.' + split[1]
    psnr_df.to_excel(psnr_path, index=0)


def test_demo(test_loader, model):
    for i in range(len(test_loader)):
        test_batch = list(test_loader)[i]
        test_batch = normalize_data_cuda(test_batch)
        input = test_batch
        with torch.no_grad():
            output = model([input, 0], mode='test')
        output = np.clip(output.cpu().numpy(), 0.0, 1.0)
        # S*B*C*H*W
        input = input[1:, 0, :, :, :]
        input = input.cpu().numpy()
        output = output[:, 0, :, :, :]  # batch=2时，只保存了一半！
        in_out = []
        for j in range(input.shape[0]):
            in_out_elem = np.concatenate((input[j, ...], output[j, ...]), axis=2)  # c h w
            in_out.append(in_out_elem)
        in_out = np.array(in_out)  # s c h w
        test_demo_save_path = os.path.join(cfg.GLOBAL.MODEL_LOG_SAVE_PATH, 'demo', 'random_seed_' + str(i + 1) + '_demo')
        if not os.path.exists(test_demo_save_path):
            os.makedirs(test_demo_save_path)
        save_movie(data=input, save_path=os.path.join(test_demo_save_path, 'truth.avi'))
        save_movie(data=output, save_path=os.path.join(test_demo_save_path, 'pred.avi'))
        save_movie(data=in_out, save_path=os.path.join(test_demo_save_path, 'truth_pred.avi'))
        save_image(data=input, save_path=os.path.join(test_demo_save_path, 'truth_img'))
        save_image(data=output, save_path=os.path.join(test_demo_save_path, 'pred_img'))
        save_image(data=in_out, save_path=os.path.join(test_demo_save_path, 'truth_pred_img'))
        print('save movies and images done!')
