import numpy as np


def dBZ_to_Pixel(dBZ):
    P = np.clip(dBZ / 60.0, a_min=0.0, a_max=1.0)
    return P


def Pixel_to_dBZ(P):
    dBZ = P * 60.0
    return dBZ


def R_to_dBZ(R):
    Z = 300 * R ** 1.4
    dBZ = 10 * np.log10(Z)
    return dBZ


def dBZ_to_R(dBZ):
    Z = 10 ** (dBZ / 10)
    R = (Z / 300) ** (1 / 1.4)
    return R


def R_to_Pixel(R):
    dBZ = R_to_dBZ(R)
    P = dBZ_to_Pixel(dBZ)
    return P


def Pixel_to_R(P):
    dBZ = Pixel_to_dBZ(P)
    R = dBZ_to_R(dBZ)
    return R
