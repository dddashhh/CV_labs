import os
import sys

import cv2
import numpy as np


def make(size=512, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 235, dtype=np.uint8)
    s = size / 512.0

    def p(x, y):
        return int(round(x * s)), int(round(y * s))

    def t(value=2):
        return max(1, int(round(value * s)))

    cv2.circle(img, p(85, 85), t(48), 35, t(3))

    cv2.circle(img, p(205, 80), t(30), 65, -1)

    cv2.circle(img, p(340, 85), t(43), 45, t(2))
    cv2.circle(img, p(340, 85), t(25), 90, t(2))
    cv2.circle(img, p(340, 85), t(7), 25, -1)

    cv2.rectangle(img, p(405, 40), p(480, 115), 30, t(3))

    cv2.rectangle(img, p(35, 155), p(125, 205), 70, -1)

    triangle = np.array(
        [p(175, 145), p(225, 215), p(125, 215)],
        dtype=np.int32,
    )
    cv2.polylines(img, [triangle], True, 25, t(3))

    triangle_filled = np.array(
        [p(300, 155), p(365, 215), p(275, 215)],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [triangle_filled], 100)

    diamond = np.array(
        [p(435, 155), p(480, 195), p(435, 235), p(390, 195)],
        dtype=np.int32,
    )
    cv2.polylines(img, [diamond], True, 40, t(2))

    cv2.ellipse(img, p(90, 265), p(55, 25), 25, 0, 360, 55, t(3))

    cv2.ellipse(img, p(220, 270), p(45, 35), 0, 180, 360, 30, t(3))

    hexagon = np.array(
        [
            p(340, 235), p(380, 255), p(380, 295),
            p(340, 315), p(300, 295), p(300, 255),
        ],
        dtype=np.int32,
    )
    cv2.polylines(img, [hexagon], True, 50, t(3))

    for i, thickness in enumerate([1, 2, 4, 6]):
        y = 335 + i * 13
        cv2.line(img, p(25, y), p(225, y), 25 + i * 20, t(thickness))

    for i, thickness in enumerate([1, 2, 3, 5]):
        x = 250 + i * 15
        cv2.line(img, p(x, 330), p(x + 5, 425), 35, t(thickness))

    cv2.line(img, p(335, 330), p(470, 425), 20, t(1))
    cv2.line(img, p(335, 345), p(470, 440), 55, t(3))
    cv2.line(img, p(335, 360), p(470, 455), 90, t(5))

    for i in range(9):
        y = 445 + i * 6
        cv2.line(img, p(25, y), p(180, y + 8), 35, t(1))

    cv2.line(img, p(200, 445), p(285, 500), 20, t(2))
    cv2.line(img, p(200, 500), p(285, 445), 20, t(2))
    cv2.line(img, p(220, 440), p(220, 505), 80, t(1))
    cv2.line(img, p(195, 470), p(290, 470), 80, t(1))

    polyline = np.array(
        [
            p(300, 450), p(320, 480), p(340, 455),
            p(360, 495), p(380, 460), p(410, 500),
            p(440, 465), p(480, 495),
        ],
        dtype=np.int32,
    )
    cv2.polylines(img, [polyline], False, 30, t(2))

    x_values = np.arange(15, 250, max(1, int(2 / s)))
    for amplitude, period, offset, intensity, thickness in [
        (12, 35, 295, 25, 2),
        (7, 22, 310, 65, 1),
        (16, 50, 285, 100, 3),
    ]:
        points = [
            p(x, offset + amplitude * np.sin(x / period * 2 * np.pi))
            for x in x_values
        ]
        cv2.polylines(
            img,
            [np.array(points, dtype=np.int32)],
            False,
            intensity,
            t(thickness),
        )

    x0, y0, x1, y1 = 365, 310, 490, 415
    tile = max(2, int(round(10 * s)))

    for y in range(int(y0 * s), int(y1 * s), tile):
        for x in range(int(x0 * s), int(x1 * s), tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                cv2.rectangle(
                    img,
                    (x, y),
                    (
                        min(x + tile - 1, int(x1 * s) - 1),
                        min(y + tile - 1, int(y1 * s) - 1),
                    ),
                    45,
                    -1,
                )

    for y in range(155, 226, 14):
        for x in range(20, 106, 14):
            cv2.circle(img, p(x, y), t(1), 45, -1)

    for x, y, radius in [
        (25, 245, 1), (45, 245, 2), (65, 245, 3),
        (85, 245, 4), (105, 245, 5),
    ]:
        cv2.circle(img, p(x, y), t(radius), 20, -1)

    noise_count = max(1, int(size * size * 0.002))
    ys = rng.integers(0, size, noise_count)
    xs = rng.integers(0, size, noise_count)
    values = rng.choice(
        np.array([0, 30, 255], dtype=np.uint8),
        size=noise_count,
    )
    img[ys, xs] = values

    return img


def make_case(name, size=512, seed=0):
    rng = np.random.default_rng(seed)

    if name == "geometry":
        return make(size, seed)

    if name == "checkerboard":
        tile = max(2, size // 32)
        yy, xx = np.indices((size, size))
        return (
            ((xx // tile + yy // tile) % 2) * 255
        ).astype(np.uint8)

    if name == "sparse":
        img = np.zeros((size, size), dtype=np.uint8)
        count = max(1, size * size // 2000)
        ys = rng.integers(0, size, count)
        xs = rng.integers(0, size, count)
        img[ys, xs] = 255
        return img

    if name == "uniform_black":
        return np.zeros((size, size), dtype=np.uint8)

    if name == "uniform_white":
        return np.full((size, size), 255, dtype=np.uint8)

    if name in (
        "dog_photo_gallai",
        "dog_photo_bluz",
        "dog_mask",
    ):
        asset_names = {
            "dog_photo_gallai": "dog_gallai.png",
            "dog_photo_bluz": "dog_bluz.png",
            "dog_mask": "dog_mask_generated.png",
        }

        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "images",
            "dogs",
            asset_names[name],
        )

        photo = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if photo is None:
            raise FileNotFoundError(
                f"Не найден тестовый файл: {path}"
            )

        h, w = photo.shape
        side = max(h, w)
        canvas = np.zeros((side, side), dtype=np.uint8)

        y, x = (side - h) // 2, (side - w) // 2
        canvas[y:y + h, x:x + w] = photo

        interpolation = (
            cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
        )
        return cv2.resize(
            canvas,
            (size, size),
            interpolation=interpolation,
        )

    raise ValueError(f"Неизвестный тип изображения: {name}")


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    out = sys.argv[2] if len(sys.argv) > 2 else "../images/test.png"

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    success = cv2.imwrite(out, make(size))
    if not success:
        raise RuntimeError(f"Не удалось сохранить изображение: {out}")

    print("saved", out)

