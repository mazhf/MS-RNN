import numpy as np
from numba import jit, float32, boolean, int32
from config import cfg
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
from util.utils import rainfall_to_pixel
from util.load_DWD import R_to_P
from util.meteo import R2Pixel
from util.brasil import R_to_Pixel


warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


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
        thresholds = cfg.HKO.THRESHOLDS
    assert 5 == prediction.ndim
    assert 5 == truth.ndim
    assert prediction.shape == truth.shape
    assert prediction.shape[2] == 1
    if cfg.dataset[0: 3] == 'HKO':
        thresholds = [rainfall_to_pixel(thresholds[i]) for i in range(len(thresholds))]
    elif cfg.dataset[0: 3] == 'DWD':
        thresholds = [R_to_P(thresholds[i]) for i in range(len(thresholds))]
    elif cfg.dataset[0: 3] == 'Met':
        thresholds = [R2Pixel(thresholds[i]) for i in range(len(thresholds))]
    elif cfg.dataset[0: 3] == 'RAI':
        thresholds = [R2Pixel(thresholds[i]) for i in range(len(thresholds))]
    elif cfg.dataset[0: 3] == 'Bra':
        thresholds = [R_to_Pixel(thresholds[i]) for i in range(len(thresholds))]
    else:
        thresholds = [rainfall_to_pixel(thresholds[i]) for i in range(len(thresholds))]
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
    if thresholds is None:
        thresholds = cfg.HKO.THRESHOLDS
    assert data.shape[2] == 1
    if cfg.dataset[0: 3] == 'HKO':
        thresholds = [rainfall_to_pixel(thresholds[i]) for i in range(len(thresholds))]
    elif cfg.dataset[0: 3] == 'DWD':
        thresholds = [R_to_P(thresholds[i]) for i in range(len(thresholds))]
    elif cfg.dataset[0: 3] == 'Met':
        thresholds = [R2Pixel(thresholds[i]) for i in range(len(thresholds))]
    elif cfg.dataset[0: 3] == 'RAI':
        thresholds = [R2Pixel(thresholds[i]) for i in range(len(thresholds))]
    elif cfg.dataset[0: 3] == 'Bra':
        thresholds = [R_to_Pixel(thresholds[i]) for i in range(len(thresholds))]
    else:
        thresholds = [rainfall_to_pixel(thresholds[i]) for i in range(len(thresholds))]
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
