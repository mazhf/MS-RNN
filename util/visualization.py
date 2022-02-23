import cv2
import numpy as np
import sys
sys.path.append("..")
from config import cfg
import os


def save_movie(data, save_path):
    seq_len, channels, height, width = data.shape
    data = data.transpose(0, 2, 3, 1)
    display_data = []
    if data.dtype == cfg.data_type:
        data = (data * 255).astype(np.uint8)
    assert data.dtype == np.uint8
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # *'I420' *'PIM1' *'XVID' *'MJPG' 视频大小越来越小，都支持.avi
    writer = cv2.VideoWriter(save_path, fourcc, 1.0, (width, height))
    for i in range(seq_len):
        if cfg.dataset in ['human3.6m', 'ucf50', 'sports10', 'deformingthings4d']:
            color_data = data[i]
        else:
            color_data = cv2.cvtColor(data[i], cv2.COLOR_GRAY2BGR)
        im = color_data
        display_data.append(im)
        writer.write(im)
    writer.release()


def save_image(data, save_path):
    data = data.transpose(0, 2, 3, 1)  # s h w c
    display_data = []
    if data.dtype == cfg.data_type:
        data = (data * 255).astype(np.uint8)
    assert data.dtype == np.uint8
    for i in range(data.shape[0]):
        if cfg.dataset in ['human3.6m', 'ucf50', 'sports10', 'deformingthings4d']:
            color_data = data[i]
        else:
            color_data = cv2.cvtColor(data[i], cv2.COLOR_GRAY2BGR)
        im = color_data
        display_data.append(im)
    arr = np.array(display_data)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(arr.shape[0]):
        cv2.imwrite(os.path.join(save_path, str(i + 1) + '.png'), arr[i, ...])

