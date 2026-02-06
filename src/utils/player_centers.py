import cv2
import torch
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max


def extract_player_centers_from_seg(
    seg_image: np.ndarray,
    offense_mask: np.ndarray,
    min_distance: int = 22,
    threshold_abs: int = 5,
):
    if offense_mask.max() > 1:
        offense_mask = (offense_mask > 0).astype(np.uint8)

    gray = cv2.cvtColor(seg_image, cv2.COLOR_RGB2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, seg_bin = cv2.threshold(
        gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    clean_mask = seg_bin * offense_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    distance = ndi.distance_transform_edt(clean_mask)
    distance[distance < threshold_abs] = 0

    centers_rc = peak_local_max(
        distance,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        labels=clean_mask,
    )

    points = np.array([[c, r] for r, c in centers_rc], dtype=np.int32)

    return points, distance

def extract_centers(heatmap, k=11, threshold=0.3):
    heatmap = heatmap.squeeze()

    peaks = local_maxima(heatmap, min_distance=12)
    peaks = [p for p in peaks if heatmap[p] > threshold]

    peaks = sorted(peaks, key=lambda p: heatmap[p], reverse=True)
    return peaks[:k]

def build_point_tensors(centers_list, device):
    B = len(centers_list)
    Nmax = max(len(c) for c in centers_list)

    points_xy = torch.zeros((B, Nmax, 2), device=device)
    points_label = torch.zeros((B, Nmax), dtype=torch.long, device=device)
    valid_mask = torch.zeros((B, Nmax), dtype=torch.bool, device=device)

    for i, centers in enumerate(centers_list):
        if len(centers) == 0:
            continue
        c = torch.tensor(centers, device=device)
        n = c.shape[0]
        points_xy[i, :n] = c
        points_label[i, :n] = 1
        valid_mask[i, :n] = True

    return points_xy, points_label, valid_mask