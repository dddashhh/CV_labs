import json
from pathlib import Path

import cv2
import numpy as np

from lab2.tracker import frame_corners


def make_video(path, count=100):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    h, w = 360, 640
    texture = rng.integers(35, 215, (h, w, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), 0)
    for i in range(120):
        x, y = rng.integers([10, 10], [w-10, h-10])
        color = tuple(map(int, rng.integers(0, 256, 3)))
        cv2.circle(texture, (int(x), int(y)), int(rng.integers(3, 14)), color, -1)
        if i % 5 == 0:
            cv2.putText(texture, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
    background = np.full_like(texture, (70, 78, 85))
    for i in range(25):
        x, y = rng.integers([0, 0], [w, h])
        cv2.line(background, (int(x), int(y)), (int(x)+25, int(y)+15), (100, 90, 65), 2)
    src = frame_corners(texture.shape)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
    if not writer.isOpened():
        raise RuntimeError('Cannot create synthetic video')
    truth = []
    try:
        for i in range(count):
            t = i / max(1, count-1)
            if i == 0:
                q = src.copy()
            else:
                scale = max(.42, 1 - i * .035) + .10 * np.sin(t * 12)
                angle = .35 * np.sin(t * 9)
                rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                q = (src - [w/2, h/2]) @ rot.T * scale + [w/2 + 45*np.sin(t*10), h/2 + 24*np.sin(t*7)]
                q += np.array([[8, -6], [-15, 8], [13, 5], [-7, -9]]) * np.sin(t*11)
                q = q.astype(np.float32)
            if 45 <= i < 53:
                frame = background.copy()
                truth.append(None)
            else:
                homography = cv2.getPerspectiveTransform(src, q)
                frame = cv2.warpPerspective(texture, homography, (w, h))
                mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), homography, (w, h))
                frame[mask == 0] = background[mask == 0]
                if 22 <= i < 35:
                    frame = cv2.GaussianBlur(frame, (13, 13), 3)
                if 65 <= i < 75:
                    cv2.rectangle(frame, (310, 130), (390, 235), (70, 78, 85), -1)
                truth.append(q.tolist())
            writer.write(frame)
    finally:
        writer.release()
    path.with_suffix('.json').write_text(json.dumps(truth))
    return truth
