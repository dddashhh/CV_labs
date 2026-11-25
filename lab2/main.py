import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from lab2.tracker import Tracker, draw


def process(video, out, mode='hybrid', corners=None, label='Object', preview=False):
    video, out = Path(video), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    writer = None
    rows = []
    elapsed = 0.0
    samples = []
    try:
        ok, first = cap.read()
        if not ok:
            raise ValueError(f'Cannot read {video}')
        h, w = first.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        writer = cv2.VideoWriter(str(out / 'tracked.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError('Cannot create output video')
        start = time.perf_counter()
        tracker = Tracker(first, corners, mode)
        elapsed += time.perf_counter() - start
        index = 0
        frame = first
        while True:
            start = time.perf_counter()
            result = tracker.previous if index == 0 else tracker.update(frame)
            elapsed += time.perf_counter() - start
            rendered = draw(frame, result, label)
            writer.write(rendered)
            row = dict(frame=index, found=int(result is not None), method=result.method if result else 'lost', inliers=result.inliers if result else 0, ratio=result.ratio if result else '', reprojection=result.error if result else '', corners=json.dumps(result.corners.tolist()) if result else '')
            rows.append(row)
            if index % max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 8)) == 0 and len(samples) < 9:
                samples.append((index, rendered.copy()))
            if preview:
                cv2.imshow('Lab 2', rendered)
                if cv2.waitKey(1) & 255 == 27:
                    break
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if preview:
            cv2.destroyAllWindows()
    with (out / 'frames.csv').open('w', newline='') as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    panel = []
    for idx, image in samples:
        thumb = cv2.resize(image, (320, int(image.shape[0] * 320 / image.shape[1])))
        cv2.putText(thumb, f'Frame {idx}', (7, thumb.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 2)
        panel.append(thumb)
    while len(panel) % 3:
        panel.append(np.zeros_like(panel[0]))
    cv2.imwrite(str(out / 'panel.jpg'), np.vstack([np.hstack(panel[i:i+3]) for i in range(0, len(panel), 3)]))
    evaluated = rows[1:] or rows
    found = [r for r in evaluated if r['found']]
    summary = dict(video=video.stem, mode=mode, frames=len(rows), evaluated=len(evaluated), found=len(found), coverage=len(found)/len(evaluated), fps=len(rows)/max(elapsed, 1e-9), mean_inliers=float(np.mean([r['inliers'] for r in found])) if found else 0.0, mean_reprojection=float(np.mean([r['reprojection'] for r in found])) if found else None)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('--out', type=Path, default=Path('lab2/results/custom'))
    parser.add_argument('--mode', choices=['sift', 'hybrid'], default='hybrid')
    parser.add_argument('--corners', nargs=8, type=float)
    parser.add_argument('--label', default='Object')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    cv2.setRNGSeed(1234)
    corners = args.corners
    if corners is None and args.video.with_suffix('.json').exists():
        corners = json.loads(args.video.with_suffix('.json').read_text())
    print(json.dumps(process(args.video, args.out, args.mode, corners, args.label, args.preview), indent=2))


if __name__ == '__main__':
    main()
