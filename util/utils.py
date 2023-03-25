import numpy as np
from collections import OrderedDict
import sys
sys.path.append("..")
from config import cfg
from torch import nn


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                 padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4], dilation=v[5])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = cfg.CONV_conv(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4], dilation=v[5])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
            elif 'prelu' in layer_name:
                layers.append(('prelu_' + layer_name, nn.PReLU()))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))


# for HKO
def pixel_to_dBZ(img):
    """
    Parameters
    ----------
    img : np.ndarray or float
    Returns
    -------
    """
    return img * 70.0 - 10.0


def dBZ_to_pixel(dBZ_img):
    """
    Parameters
    ----------
    dBZ_img : np.ndarray
    Returns
    -------
    """
    return np.clip((dBZ_img + 10.0) / 70.0, a_min=0.0, a_max=1.0)


def pixel_to_rainfall(img, a=58.53, b=1.56):
    """Convert the pixel values to real rainfall intensity
    Parameters
    ----------
    img : np.ndarray
    a : cfg.GLOBAL.DATA_TYPE, optional
    b : cfg.GLOBAL.DATA_TYPE, optional
    Returns
    -------
    rainfall_intensity : np.ndarray
    """
    dBZ = pixel_to_dBZ(img)
    dBR = (dBZ - 10.0 * np.log10(a)) / b
    rainfall_intensity = np.power(10, dBR / 10.0)
    return rainfall_intensity


def rainfall_to_pixel(rainfall_intensity, a=58.53, b=1.56):
    """Convert the rainfall intensity to pixel values
    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : cfg.GLOBAL.DATA_TYPE, optional
    b : cfg.GLOBAL.DATA_TYPE, optional
    Returns
    -------
    pixel_vals : np.ndarray
    """
    dBR = np.log10(rainfall_intensity) * 10.0
    # dBZ = 10b log(R) +10log(a)
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 70.0
    return pixel_vals


def dBZ_to_rainfall(dBZ, a=58.53, b=1.56):
    return np.power(10, (dBZ - 10 * np.log10(a)) / (10 * b))


def rainfall_to_dBZ(rainfall, a=58.53, b=1.56):
    return 10 * np.log10(a) + 10 * b * np.log10(rainfall)

