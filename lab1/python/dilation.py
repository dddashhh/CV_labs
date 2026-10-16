import cv2
import numpy as np
from PIL import Image, ImageFilter


def binarize(gray: np.ndarray, thresh: int = -1) -> np.ndarray:
    if thresh < 0:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    return bw


def otsu_threshold_native(img: np.ndarray) -> int:
    hist = [0] * 256
    for row in img.tolist():
        for v in row:
            hist[v] += 1
    total = sum(hist)
    sum_all = sum(i * h for i, h in enumerate(hist))
    w0 = 0
    sum0 = 0.0
    best_var, best_t = -1.0, 0
    for t in range(256):
        w0 += hist[t]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += t * hist[t]
        m0 = sum0 / w0
        m1 = (sum_all - sum0) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def binarize_native(gray: np.ndarray, thresh: int = -1) -> np.ndarray:
    if thresh < 0:
        thresh = otsu_threshold_native(gray)
    out = [[255 if v <= thresh else 0 for v in row] for row in gray.tolist()]
    return np.array(out, dtype=np.uint8)


def dilate_native(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    src = img.tolist()
    dst = [[0] * w for _ in range(h)]
    for y in range(h):
        y0 = y - 1 if y > 0 else 0
        y1 = y + 1 if y < h - 1 else h - 1
        for x in range(w):
            x0 = x - 1 if x > 0 else 0
            x1 = x + 1 if x < w - 1 else w - 1
            m = 0
            for yy in range(y0, y1 + 1):
                row = src[yy]
                for xx in range(x0, x1 + 1):
                    if row[xx] > m:
                        m = row[xx]
            dst[y][x] = m
    return np.array(dst, dtype=np.uint8)


_KERNEL = np.ones((3, 3), np.uint8)


def dilate_opencv(img: np.ndarray) -> np.ndarray:
    return cv2.dilate(img, _KERNEL, iterations=1)


def dilate_pil(img: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(img)
    return np.asarray(pil.filter(ImageFilter.MaxFilter(3)))
