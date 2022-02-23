import numpy as np
import torch
from numba import jit, float32, boolean, int32
from util.utils import rainfall_to_pixel
from util.utils import R_to_P
from config import cfg
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
import cv2
from scipy.spatial.distance import pdist

warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


@jit(float32(float32))
def compute_focal_numba(differ):
    S, B, C, H, W = differ.shape
    differ = np.abs(differ)
    differ = differ.transpose(1, 0, 2, 3, 4)  # b s c h w
    focal_pixels_matrix = np.ones_like(differ)
    focal_frames_matrix = np.ones((B, S))
    for b in range(B):
        sum_frames = np.sum(differ[b])  # sum s c h w
        for s in range(S):
            sum_pixels = np.sum(differ[b, s])  # sum c h w
            focal_frames_matrix[b, s] = sum_pixels / sum_frames  # b s
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        focal_pixels_matrix[b, s, c, h, w] = differ[b, s, c, h, w] / sum_pixels
    return focal_pixels_matrix.transpose(1, 0, 2, 3, 4), focal_frames_matrix.transpose(1, 0)


@jit(float32(float32, float32))
def compute_MMD_distance_numba(X, Y):
    S, B, C, H, W = X.shape
    X = X.reshape(S, B, C, H * W)
    Y = Y.reshape(S, B, C, H * W)
    X = X.permute(1, 2, 0, 3)  # B, C, S, H * W
    Y = Y.permute(1, 2, 0, 3)  # B, C, S, H * W
    kxx = 0
    kyy = 0
    kxy = 0
    for b in range(B):
        for c in range(C):
            for i in range(S):
                for j in range(S):
                    if i != j:
                        xi = X[b, c, i]
                        xj = X[b, c, j]
                        kxx = kxx + (xi * xj.T + 1) ** 3 / (S * (S - 1))
                        yi = Y[b, c, i]
                        yj = Y[b, c, j]
                        kyy = kyy + (yi * yj.T + 1) ** 3 / (S * (S - 1))
                    xi = X[b, c, i]
                    yj = Y[b, c, j]
                    kxy = kxy + (xi * yj.T + 1) ** 3 / S ** 2
    mmd = (kxx - 2 * kxy + kyy) / (B * C)
    return mmd


@jit(float32(float32, float32, float32))
def compute_hamming_distance_numba(truth, pred, pixel_weights):
    s, b, c, h, w = truth.shape
    for i in range(s):
        for j in range(b):
            for k in range(c):
                for m in range(h):
                    for n in range(w):
                        if truth[i, j, k, m, n] != 0:
                            truth[i, j, k, m, n] = 1
                        if pred[i, j, k, m, n] != 0:
                            pred[i, j, k, m, n] = 1
    truth = truth.flatten()
    pred = pred.flatten()
    pixel_weights = pixel_weights.flatten()
    hamming = pdist(np.vstack([truth, pred]), metric='hamming', w=pixel_weights)
    return hamming


@jit(float32(float32, float32, float32))
def compute_cosine_distance_numba(truth, pred, pixel_weights):
    s, b, c, h, w = truth.shape
    truth = truth.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    pixel_weights = pixel_weights.detach().cpu().numpy()
    # for i in range(s):
    #     for j in range(b):
    #         for k in range(c):
    #             for m in range(h):
    #                 for n in range(w):
    #                     if truth[i, j, k, m, n] != 0:
    #                         truth[i, j, k, m, n] = 1
    #                     if pred[i, j, k, m, n] != 0:
    #                         pred[i, j, k, m, n] = 1
    truth = truth.flatten()
    pred = pred.flatten()
    pixel_weights = pixel_weights.flatten()
    cosine = pdist(np.vstack([truth, pred]), metric='cosine')
    # cosine = pdist(np.vstack([truth, pred]), metric='cosine', w=pixel_weights)
    cosine = torch.Tensor(cosine).cuda()
    return cosine


@jit(int32(float32, float32, float32))
def count_identify_pixel_numba(truth, pred, pixel_weights):
    s, b, c, h, w = truth.shape
    count = 0
    for i in range(s):
        for j in range(b):
            for k in range(c):
                for m in range(h):
                    for n in range(w):
                        if truth[i, j, k, m, n] == pred[i, j, k, m, n]:
                            count += pixel_weights[i, j, k, m, n]
    return count


@jit(float32(float32, int32, int32))
def resize_numba(tensor, h_new, w_new):
    # 对于tensor不可用，但是可以对普通数组操作时做个参考
    s_b, c, h, w = tensor.shape
    for i in range(s_b):
        for j in range(c):
            tensor[i, j] = cv2.resize(tensor[i, j], (h_new, w_new))
    return tensor


@jit(float32(float32, float32))
def get_GDL_numba(prediction, truth):
    """Accelerated version of get_GDL using numba(http://numba.pydata.org/)
    """
    seq_len, batch_size, _, height, width = prediction.shape
    gdl = np.zeros(shape=(seq_len, batch_size), dtype=cfg.data_type)
    for i in range(seq_len):
        for j in range(batch_size):
            for m in range(height):
                for n in range(width):
                    if m + 1 < height:
                        pred_diff_h = abs(prediction[i][j][0][m + 1][n] - prediction[i][j][0][m][n])
                        gt_diff_h = abs(truth[i][j][0][m + 1][n] - truth[i][j][0][m][n])
                        gdl[i][j] += abs(pred_diff_h - gt_diff_h)
                    if n + 1 < width:
                        pred_diff_w = abs(prediction[i][j][0][m][n + 1] - prediction[i][j][0][m][n])
                        gt_diff_w = abs(truth[i][j][0][m][n + 1] - truth[i][j][0][m][n])
                        gdl[i][j] += abs(pred_diff_w - gt_diff_w)
    return gdl


def get_hit_miss_counts_numba(prediction, truth, thresholds=None):
    """This function calculates the overall hits and misses for the prediction, which could be used
    to get the skill scores and threat scores:

    This function assumes the input, i.e, prediction and truth are 3-dim tensors, (timestep, row, col)
    and all inputs should be between 0~1

    Parameters
    ----------
    prediction : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    truth : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    thresholds : list or tuple

    Returns
    -------
    hits : np.ndarray
        (seq_len, batch_size, len(thresholds))
        TP
    misses : np.ndarray
        (seq_len, batch_size, len(thresholds))
        FN
    false_alarms : np.ndarray
        (seq_len, batch_size, len(thresholds))
        FP
    correct_negatives : np.ndarray
        (seq_len, batch_size, len(thresholds))
        TN
    """
    if thresholds is None:
        thresholds = cfg.HKO.EVALUATION.THRESHOLDS
    assert 5 == prediction.ndim
    assert 5 == truth.ndim
    assert prediction.shape == truth.shape
    assert prediction.shape[2] == 1
    if cfg.dataset[0: 3] == 'HKO':
        thresholds = [rainfall_to_pixel(thresholds[i]) for i in range(len(thresholds))]
    elif cfg.dataset[0: 3] == 'DWD':
        thresholds = [R_to_P(thresholds[i]) for i in range(len(thresholds))]
    thresholds = sorted(thresholds)
    ret = _get_hit_miss_counts_numba(truth=truth, prediction=prediction, thresholds=thresholds)
    return ret[:, :, :, 0], ret[:, :, :, 1], ret[:, :, :, 2], ret[:, :, :, 3]


@jit(int32(float32, float32, float32))
def _get_hit_miss_counts_numba(truth, prediction, thresholds):
    seq_len, batch_size, _, height, width = prediction.shape
    threshold_num = len(thresholds)
    ret = np.zeros(shape=(seq_len, batch_size, threshold_num, 4), dtype=np.int32)

    for i in range(seq_len):
        for j in range(batch_size):
            for m in range(height):
                for n in range(width):
                    for k in range(threshold_num):
                        bpred = prediction[i][j][0][m][n] >= thresholds[k]  # 1 or 0
                        btruth = truth[i][j][0][m][n] >= thresholds[k]  # 1 or 0
                        ind = (1 - btruth) * 2 + (1 - bpred)
                        ret[i][j][k][ind] += 1
                        # The above code is the same as:
                        # btruth和bpred的四种情况，对应某个像素点出现的四种情况
                        # TP
                        # ret[i][j][k][0] += bpred * btruth
                        # FP
                        # ret[i][j][k][1] += (1 - bpred) * btruth
                        # TN
                        # ret[i][j][k][2] += bpred * (1 - btruth)
                        # FN
                        # ret[i][j][k][3] += (1 - bpred) * (1- btruth)
    return ret


def get_balancing_weights_numba(data, base_balancing_weights=None, thresholds=None):
    assert data.shape[2] == 1
    if cfg.dataset[0: 3] == 'HKO':
        thresholds = [rainfall_to_pixel(thresholds[i]) for i in range(len(thresholds))]
    elif cfg.dataset[0: 3] == 'DWD':
        thresholds = [R_to_P(thresholds[i]) for i in range(len(thresholds))]
    thresholds = sorted(thresholds)
    ret = _get_balancing_weights_numba(data=data, base_balancing_weights=base_balancing_weights,
                                       thresholds=thresholds)
    return ret


@jit(float32(float32, float32, float32))
def _get_balancing_weights_numba(data, base_balancing_weights, thresholds):
    seq_len, batch_size, _, height, width = data.shape
    threshold_num = len(thresholds)
    ret = np.zeros(shape=(seq_len, batch_size, 1, height, width), dtype=cfg.data_type)

    for i in range(seq_len):
        for j in range(batch_size):
            for m in range(height):
                for n in range(width):
                    ele = data[i][j][0][m][n]
                    for k in range(threshold_num):
                        if ele < thresholds[k]:
                            ret[i][j][0][m][n] = base_balancing_weights[k]
                            break
                    if ele >= thresholds[threshold_num - 1]:
                        ret[i][j][0][m][n] = base_balancing_weights[threshold_num]
    return ret  # weight s*b*1*h*w
