import numpy as np
from scipy.fft import dctn, idctn

def dct_features(gray_img):
    """
    DCT - Discrete Cosine Transform Feature Extraction
    Applies 2D-DCT and extracts energy distribution across frequency bands.
    Also reconstructs image from low-frequency coefficients only.
    """
    img = gray_img.astype(np.float64)
    rows, cols = img.shape

    # 2D DCT
    dct_matrix = dctn(img, norm='ortho')

    # Energy in frequency bands
    total_energy = np.sum(dct_matrix ** 2)

    r8  = min(8,  rows); c8  = min(8,  cols)
    r32 = min(32, rows); c32 = min(32, cols)

    low_freq_energy  = np.sum(dct_matrix[:r8,  :c8 ] ** 2)
    mid_freq_energy  = np.sum(dct_matrix[r8:r32, r8:c32] ** 2)
    high_freq_energy = np.sum(dct_matrix[r32:, r32:] ** 2)

    # Top 8x8 coefficients
    top_coeffs = dct_matrix[:r8, :c8]

    # Reconstruct from low-freq only (compression demo)
    dct_low = np.zeros_like(dct_matrix)
    dct_low[:r32, :c32] = dct_matrix[:r32, :c32]
    reconstructed = np.clip(idctn(dct_low, norm='ortho'), 0, 255).astype(np.uint8)

    features = {
        'dct_matrix':       dct_matrix,
        'top_coeffs':       top_coeffs,
        'total_energy':     total_energy,
        'low_freq_energy':  low_freq_energy,
        'mid_freq_energy':  mid_freq_energy,
        'high_freq_energy': high_freq_energy,
        'reconstructed':    reconstructed
    }

    print("\n===== DCT Features =====")
    print(f"Total Energy:     {total_energy:.2e}")
    if total_energy > 0:
        print(f"Low Freq Energy:  {100*low_freq_energy/total_energy:.2f}%")
        print(f"Mid Freq Energy:  {100*mid_freq_energy/total_energy:.2f}%")
        print(f"High Freq Energy: {100*high_freq_energy/total_energy:.2f}%")

    return features
