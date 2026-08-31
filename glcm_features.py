import numpy as np

def glcm_features(gray_img):
    """
    GLCM - Gray Level Co-occurrence Matrix Feature Extraction
    Extracts: Contrast, Correlation, Energy, Homogeneity
    Averaged over 4 directions: 0°, 45°, 90°, 135°
    """
    # Reduce to 64 gray levels for performance
    img = (gray_img.astype(np.float64) / 255.0 * 63).astype(np.uint8)
    levels = 64
    rows, cols = img.shape

    # 4 direction offsets: 0°, 45°, 90°, 135°
    offsets = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]
    G_sum = np.zeros((levels, levels), dtype=np.float64)

    for dy, dx in offsets:
        G = np.zeros((levels, levels), dtype=np.float64)
        for r in range(max(0, -dy), min(rows, rows - dy)):
            for c in range(max(0, -dx), min(cols, cols - dx)):
                i = img[r, c]
                j = img[r + dy, c + dx]
                G[i, j] += 1
                G[j, i] += 1  # symmetric
        G_sum += G

    # Normalize
    G_avg = G_sum / G_sum.sum()

    # Build index grids
    i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')

    # Contrast
    contrast = np.sum((i_idx - j_idx) ** 2 * G_avg)

    # Correlation
    mu_i = np.sum(i_idx * G_avg)
    mu_j = np.sum(j_idx * G_avg)
    sig_i = np.sqrt(np.sum((i_idx - mu_i) ** 2 * G_avg))
    sig_j = np.sqrt(np.sum((j_idx - mu_j) ** 2 * G_avg))
    if sig_i * sig_j == 0:
        correlation = 0.0
    else:
        correlation = np.sum((i_idx - mu_i) * (j_idx - mu_j) * G_avg) / (sig_i * sig_j)

    # Energy (Angular Second Moment)
    energy = np.sum(G_avg ** 2)

    # Homogeneity (Inverse Difference Moment)
    homogeneity = np.sum(G_avg / (1 + np.abs(i_idx - j_idx)))

    features = {
        'contrast':    contrast,
        'correlation': correlation,
        'energy':      energy,
        'homogeneity': homogeneity,
        'matrix':      G_avg
    }

    print("\n===== GLCM Features =====")
    print(f"Contrast:    {contrast:.4f}")
    print(f"Correlation: {correlation:.4f}")
    print(f"Energy:      {energy:.4f}")
    print(f"Homogeneity: {homogeneity:.4f}")

    return features
