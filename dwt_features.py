import numpy as np
from scipy.signal import convolve2d

def haar_dwt2(img):
    """Single-level 2D Haar DWT decomposition."""
    lp = np.array([1, 1]) / np.sqrt(2)
    hp = np.array([1, -1]) / np.sqrt(2)

    # Filter rows
    L_row = convolve2d(img, lp[np.newaxis, :], mode='same')
    H_row = convolve2d(img, hp[np.newaxis, :], mode='same')

    # Filter cols + downsample
    LL = convolve2d(L_row, lp[:, np.newaxis], mode='same')[::2, ::2]
    LH = convolve2d(L_row, hp[:, np.newaxis], mode='same')[::2, ::2]
    HL = convolve2d(H_row, lp[:, np.newaxis], mode='same')[::2, ::2]
    HH = convolve2d(H_row, hp[:, np.newaxis], mode='same')[::2, ::2]

    return LL, LH, HL, HH


def dwt_features(gray_img):
    """
    DWT - Discrete Wavelet Transform Feature Extraction
    Uses Haar wavelet, 3-level decomposition.
    Subbands: LL (approx), LH (horizontal), HL (vertical), HH (diagonal)
    """
    img = gray_img.astype(np.float64)

    energy = lambda x: np.sum(x ** 2) / x.size

    # Level 1
    LL1, LH1, HL1, HH1 = haar_dwt2(img)
    # Level 2
    LL2, LH2, HL2, HH2 = haar_dwt2(LL1)
    # Level 3
    LL3, LH3, HL3, HH3 = haar_dwt2(LL2)

    features = {
        'LL1': LL1, 'LH1': LH1, 'HL1': HL1, 'HH1': HH1,
        'LL2': LL2, 'LH2': LH2, 'HL2': HL2, 'HH2': HH2,
        'LL3': LL3, 'LH3': LH3, 'HL3': HL3, 'HH3': HH3,
        'energy_L1': [energy(LL1), energy(LH1), energy(HL1), energy(HH1)],
        'energy_L2': [energy(LL2), energy(LH2), energy(HL2), energy(HH2)],
        'energy_L3': [energy(LL3), energy(LH3), energy(HL3), energy(HH3)],
    }

    print("\n===== DWT Features (Haar, 3-level) =====")
    print("Level 1 Energy -> LL:{:.2f}  LH:{:.2f}  HL:{:.2f}  HH:{:.2f}".format(*features['energy_L1']))
    print("Level 2 Energy -> LL:{:.2f}  LH:{:.2f}  HL:{:.2f}  HH:{:.2f}".format(*features['energy_L2']))
    print("Level 3 Energy -> LL:{:.2f}  LH:{:.2f}  HL:{:.2f}  HH:{:.2f}".format(*features['energy_L3']))

    return features
