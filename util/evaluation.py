try:
    import cPickle as pickle
except:
    import pickle
from util.msssim import _SSIMForMultiScale
from util.numba_accelerated import get_GDL_numba, get_hit_miss_counts_numba, get_balancing_weights_numba
import numpy as np
import sys
sys.path.append("..")
from config import cfg
import cv2


def get_PSNR(prediction, truth):
    mse = np.square(prediction - truth).mean(axis=(2, 3, 4))
    eps = 1e-5
    ret = 10.0 * np.log10(1.0 / (mse + eps))
    return ret


def get_SSIM(prediction, truth):
    """Calculate the SSIM score following
    [TIP2004] Image Quality Assessment: From Error Visibility to Structural Similarity
    Same functionality as
    https://github.com/coupriec/VideoPredictionICLR2016/blob/master/image_error_measures.lua#L50-L75
    We use nowcasting.helpers.msssim, which is borrowed from Tensorflow to do the evaluation
    Parameters
    ----------
    prediction : np.ndarray  sbchw
    truth : np.ndarray
    Returns
    -------
    ret : np.ndarray
    :param truth:
    :param prediction:
    """
    s, b, c, h, w = prediction.shape
    prediction = prediction.reshape((s * b, h, w, c))
    truth = truth.reshape((s * b, h, w, c))
    # if 'human3.6m' in cfg.GLOBAL.dataset:
    #     for i in range(s * b):
    #         prediction[i] = cv2.cvtColor(prediction[i], cv2.COLOR_BGR2GRAY)
    #         truth[i] = cv2.cvtColor(truth[i], cv2.COLOR_BGR2GRAY)
    #     prediction = prediction[:, np.newaxis]
    #     truth = truth[:, np.newaxis]

    ssim, cs = _SSIMForMultiScale(img1=prediction, img2=truth, max_val=1.0)
    ret = ssim.reshape((s, b))
    return ret


def sum_batch(data):
    return data.sum(axis=1)


def as_type(data):
    return data.astype(cfg.data_type)


class Evaluation(object):
    def __init__(self, seq_len, use_central=False, thresholds=None):
        self._total_batch_num = 0
        self._thresholds = cfg.HKO.THRESHOLDS if thresholds is None else thresholds
        self._ssim = np.zeros((seq_len,), dtype=cfg.data_type)
        self._psnr = np.zeros((seq_len,), dtype=cfg.data_type)
        self._gdl = np.zeros((seq_len,), dtype=cfg.data_type)
        self._balanced_mae = np.zeros((seq_len,), dtype=cfg.data_type)
        self._balanced_mse = np.zeros((seq_len,), dtype=cfg.data_type)
        self._mae = np.zeros((seq_len,), dtype=cfg.data_type)
        self._mse = np.zeros((seq_len,), dtype=cfg.data_type)
        self._total_correct_negatives = np.zeros((seq_len, len(self._thresholds)), dtype=np.int32)
        self._total_false_alarms = np.zeros((seq_len, len(self._thresholds)), dtype=np.int32)
        self._total_misses = np.zeros((seq_len, len(self._thresholds)), dtype=np.int32)
        self._total_hits = np.zeros((seq_len, len(self._thresholds)), dtype=np.int32)
        self._seq_len = seq_len
        self._use_central = use_central

    def clear_all(self):
        self._total_batch_num = 0
        self._ssim[:] = 0
        self._psnr[:] = 0
        self._gdl[:] = 0
        self._balanced_mse[:] = 0
        self._balanced_mae[:] = 0
        self._mse[:] = 0
        self._mae[:] = 0
        self._total_hits[:] = 0
        self._total_misses[:] = 0
        self._total_false_alarms[:] = 0
        self._total_correct_negatives[:] = 0

    def update(self, gt, pred):
        if cfg.dataset in ['human3.6m', 'ucf50', 'sports10', 'deformingthings4d']:
            s, b, c, h, w = pred.shape
            pred = pred.reshape((s * b, h, w, c))
            gt = gt.reshape((s * b, h, w, c))
            pred_ = np.zeros((s * b, h, w))
            gt_ = np.zeros((s * b, h, w))
            for i in range(s * b):
                pred_[i, ...] = cv2.cvtColor(pred[i, ...], cv2.COLOR_BGR2GRAY)
                gt_[i, ...] = cv2.cvtColor(gt[i], cv2.COLOR_BGR2GRAY)
            pred = pred_[:, :, :, np.newaxis]
            gt = gt_[:, :, :, np.newaxis]
            pred = pred.reshape((s, b, 1, h, w))
            gt = gt.reshape((s, b, 1, h, w))
        batch_size = gt.shape[1]
        assert gt.shape[0] == self._seq_len
        assert gt.shape == pred.shape

        if self._use_central:
            # Crop the central regions for evaluation
            central_region = cfg.HKO.CENTRAL_REGION
            pred = pred[:, :, :, central_region[1]:central_region[3], central_region[0]:central_region[2]]
            gt = gt[:, :, :, central_region[1]:central_region[3], central_region[0]:central_region[2]]

        self._total_batch_num += batch_size

        # 表示需要做而未做的一些待完成的事项，有助于事后的检索，以及对整体项目做进一步的修改迭代。
        # TODO Save all the mse, mae, gdl, hits, misses, false_alarms and correct_negatives
        ssim = get_SSIM(prediction=pred, truth=gt)
        psnr = get_PSNR(prediction=pred, truth=gt)
        gdl = get_GDL_numba(prediction=pred, truth=gt)
        bw = cfg.HKO.BALANCING_WEIGHTS
        weights = get_balancing_weights_numba(data=gt, base_balancing_weights=bw, thresholds=self._thresholds)

        # S*B*1*H*W
        balanced_mse = (weights * np.square(pred - gt)).sum(axis=(2, 3, 4))
        balanced_mae = (weights * np.abs(pred - gt)).sum(axis=(2, 3, 4))
        mse = np.square(pred - gt).sum(axis=(2, 3, 4))
        mae = np.abs(pred - gt).sum(axis=(2, 3, 4))
        hits, misses, false_alarms, correct_negatives = get_hit_miss_counts_numba(prediction=pred, truth=gt,
                                                                                  thresholds=self._thresholds)
        self._gdl += sum_batch(gdl)
        self._ssim += sum_batch(ssim)
        self._psnr += sum_batch(psnr)
        self._balanced_mse += sum_batch(balanced_mse)
        self._balanced_mae += sum_batch(balanced_mae)
        self._mse += sum_batch(mse)  # s*1 batch求和，再除以batch，再求平均值（为了求最后一帧的度量，因此很麻烦），和loss中的mse一致，已验证
        self._mae += sum_batch(mae)
        self._total_hits += sum_batch(hits)
        self._total_misses += sum_batch(misses)
        self._total_false_alarms += sum_batch(false_alarms)
        self._total_correct_negatives += sum_batch(correct_negatives)

    def get_metrics(self):
        """The following measurements will be used to measure the score of the forecaster
        See Also
        [Weather and Forecasting 2010] Equitability Revisited: Why the "Equitable Threat Score" Is Not Equitable
        http://www.wxonline.info/topics/verif2.html
        We will denote
        (a b    (hits       false alarms
         c d) =  misses   correct negatives)
        We will report the
        POD = hits / (hits + misses)
        FAR = false alarms / (hits + false alarms)
        CSI = hits / (hits + false alarms + misses)
        Heidke Skill Score (HSS) = 2(ad - bc) / ((a+c) (c+d) + (a+b)(b+d))
        Gilbert Skill Score (GSS) = HSS / (2 - HSS), also known as the Equitable Threat Score
            HSS = 2 * GSS / (GSS + 1)
        MSE = (pred - gt) **2
        MAE = abs(pred - gt)
        GDL = abs(gd_h(pred) - gd_h(gt)) + abs(gd_w(pred) - gd_w(gt))
        Returns
        ssim, psnr, gdl, balanced_mse, balanced_mae, mse, mae: (seq_len, len(thresholds))
        pod, far, csi, hss: (seq_len, len(thresholds))
        -------
        """
        ssim = self._ssim / self._total_batch_num
        psnr = self._psnr / self._total_batch_num
        gdl = self._gdl / self._total_batch_num
        balanced_mse = self._balanced_mse / self._total_batch_num
        balanced_mae = self._balanced_mae / self._total_batch_num
        mse = self._mse / self._total_batch_num
        mae = self._mae / self._total_batch_num
        hits = as_type(self._total_hits)
        misses = as_type(self._total_misses)
        false_alarms = as_type(self._total_false_alarms)
        correct_negatives = as_type(self._total_correct_negatives)
        pod = hits / (hits + misses)
        far = false_alarms / (hits + false_alarms)
        csi = hits / (hits + misses + false_alarms)
        hss = 2 * (hits * correct_negatives - misses * false_alarms) / \
              ((hits + misses) * (misses + correct_negatives) +
               (hits + false_alarms) * (false_alarms + correct_negatives))
        l_all = [ssim, psnr, gdl, balanced_mse, balanced_mae, mse, mae, pod, far, csi, hss]
        return l_all
