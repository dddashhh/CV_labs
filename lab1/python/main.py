import argparse
import os

import cv2
import numpy as np

from dilation import binarize, dilate_native, dilate_opencv, dilate_pil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--thresh", type=int, default=-1, help="порог; -1 = Оцу")
    ap.add_argument("--out", default="../results")
    a = ap.parse_args()

    gray = cv2.imread(a.input, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"Не удалось открыть {a.input}")
    os.makedirs(a.out, exist_ok=True)
    bw = binarize(gray, a.thresh)
    res = {"native": dilate_native(bw), "opencv": dilate_opencv(bw), "pil": dilate_pil(bw)}
    cv2.imwrite(os.path.join(a.out, "py_binary.png"), bw)
    for k, v in res.items():
        cv2.imwrite(os.path.join(a.out, f"py_dilated_{k}.png"), v)
        print(f"{k}: совпадает с OpenCV = {np.array_equal(v, res['opencv'])}")


if __name__ == "__main__":
    main()
