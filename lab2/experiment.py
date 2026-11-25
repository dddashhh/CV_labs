import argparse
import csv
import json
from pathlib import Path

import cv2

from lab2.main import process
from lab2.report import generate_report


VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.m4v', '.mkv'}
REFERENCE_NAMES = ('mona-lisa', 'mona-lisa-blur', 'mona-lisa-blur-extra-credit')


def video_cases(videos, own_video=None, own_corners=None):
    cases = []
    paths = sorted((p for p in videos.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES), key=lambda p: (REFERENCE_NAMES.index(p.stem.lower()) if p.stem.lower() in REFERENCE_NAMES else len(REFERENCE_NAMES), p.name.lower()))
    if own_video is not None:
        if not own_video.is_file():
            raise FileNotFoundError(own_video)
        if own_video.resolve() not in {p.resolve() for p in paths}:
            paths.append(own_video)
    used = set()
    known = set()
    for path in paths:
        if path.resolve() in known:
            continue
        name = path.stem
        if name.lower() in used:
            raise ValueError(f'Duplicate video name: {path.name}')
        used.add(name.lower())
        known.add(path.resolve())
        sidecar = path.with_suffix('.json')
        corners = json.loads(sidecar.read_text()) if sidecar.exists() else None
        if own_video is not None and path.resolve() == own_video.resolve() and own_corners is not None:
            corners = own_corners
        elif path.name.lower() == 'keyboard.mp4' and corners is None:
            corners = [131, 148, 1060, 148, 1057, 539, 112, 536]
        label = 'Mona Lisa' if name.lower() in REFERENCE_NAMES else 'Keyboard' if path.name.lower() == 'keyboard.mp4' else 'Object'
        cases.append((name, path, corners, label))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', type=Path, default=Path('lab2/videos'))
    parser.add_argument('--out', type=Path, default=Path('lab2/results'))
    parser.add_argument('--own-video', type=Path)
    parser.add_argument('--own-corners', nargs=8, type=float)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    cv2.setRNGSeed(1234)
    args.out.mkdir(parents=True, exist_ok=True)
    cases = video_cases(args.videos, args.own_video, args.own_corners)
    if not cases:
        parser.error(f'No videos found in {args.videos}')
    results = []
    for name, path, corners, label in cases:
        for mode in ('sift', 'hybrid'):
            cv2.setRNGSeed(1234)
            result = process(path, args.out / name / mode, mode, corners, label)
            result['case'] = name
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    fields = list(dict.fromkeys(k for row in results for k in row))
    with (args.out / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    (args.out / 'report.md').write_text(generate_report(args.out), encoding='utf-8')


if __name__ == '__main__':
    main()
