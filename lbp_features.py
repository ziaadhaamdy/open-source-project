import numpy as np

def lbp_features(gray_img):
    """
    LBP - Local Binary Pattern Feature Extraction
    Uses circular (8,1) LBP — 8 neighbors, radius 1.
    """
    img = gray_img.astype(np.float64)
    rows, cols = img.shape
    lbp_map = np.zeros((rows, cols), dtype=np.uint8)

    # 8-neighbor circular offsets for radius=1
    angles = np.arange(8) * (2 * np.pi / 8)
    dx = np.round(np.cos(angles)).astype(int)
    dy = np.round(-np.sin(angles)).astype(int)

    # Compute LBP for interior pixels
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            center = img[r, c]
            code = 0
            for k in range(8):
                neighbor = img[r + dy[k], c + dx[k]]
                if neighbor >= center:
                    code += 2 ** k
            lbp_map[r, c] = code

    # LBP histogram (256 bins), normalized
    hist, _ = np.histogram(lbp_map.ravel(), bins=256, range=(0, 256))
    histogram = hist.astype(np.float64) / lbp_map.size

    # Descriptors
    uniformity   = np.sum(histogram ** 2)
    nonzero_hist = histogram[histogram > 0]
    entropy      = -np.sum(nonzero_hist * np.log2(nonzero_hist))
    mean_pattern = np.mean(lbp_map.astype(np.float64))
    std_pattern  = np.std(lbp_map.astype(np.float64))

    features = {
        'lbp_map':      lbp_map,
        'histogram':    histogram,
        'uniformity':   uniformity,
        'entropy':      entropy,
        'mean_pattern': mean_pattern,
        'std_pattern':  std_pattern,
    }

    print("\n===== LBP Features =====")
    print(f"Uniformity (Energy): {uniformity:.4f}")
    print(f"Entropy:             {entropy:.4f} bits")
    print(f"Mean Pattern:        {mean_pattern:.2f}")
    print(f"Std Pattern:         {std_pattern:.2f}")

    return features
