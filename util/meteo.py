import numpy as np


def dBZ2Pixel(dBZ):
    P = np.clip(dBZ / 70.0, a_min=0.0, a_max=1.0)
    return P


def Pixel2dBZ(P):
    dBZ = P * 70.0
    return dBZ


def R2dBZ(R):
    Z = 200 * R ** 1.6
    dBZ = 10 * np.log10(Z)
    return dBZ


def dBZ2R(dBZ):
    Z = 10 ** (dBZ / 10)
    R = (Z / 200) ** (1 / 1.6)
    return R


def R2Pixel(R):
    dBZ = R2dBZ(R)
    P = dBZ2Pixel(dBZ)
    return P


def Pixel2R(P):
    dBZ = Pixel2dBZ(P)
    R = dBZ2R(dBZ)
    return R
