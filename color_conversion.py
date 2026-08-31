import numpy as np

def rgb_to_gray(img):
    """
    Convert RGB image to Grayscale using ITU-R BT.601:
    Gray = 0.299*R + 0.587*G + 0.114*B
    """
    R = img[:, :, 0].astype(np.float64)
    G = img[:, :, 1].astype(np.float64)
    B = img[:, :, 2].astype(np.float64)
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return np.clip(gray, 0, 255).astype(np.uint8)


def rgb_to_ycbcr(img):
    """
    Convert RGB image to YCbCr using ITU-R BT.601:
    Y  =  16 + 65.481*R + 128.553*G + 24.966*B
    Cb = 128 - 37.797*R -  74.203*G + 112.000*B
    Cr = 128 + 112.000*R - 93.786*G -  18.214*B
    (R, G, B normalized to [0, 1])
    """
    R = img[:, :, 0].astype(np.float64) / 255.0
    G = img[:, :, 1].astype(np.float64) / 255.0
    B = img[:, :, 2].astype(np.float64) / 255.0

    Y  =  16  + 65.481 * R + 128.553 * G +  24.966 * B
    Cb = 128  - 37.797 * R -  74.203 * G + 112.000 * B
    Cr = 128  + 112.000 * R - 93.786 * G -  18.214 * B

    Y  = np.clip(Y,  0, 255).astype(np.uint8)
    Cb = np.clip(Cb, 0, 255).astype(np.uint8)
    Cr = np.clip(Cr, 0, 255).astype(np.uint8)

    return np.stack([Y, Cb, Cr], axis=2)
