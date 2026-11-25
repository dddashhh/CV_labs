import csv
from pathlib import Path

import cv2
import pytest

from lab2.main import process
from lab2.report import generate_report


def test_real_video_pipeline(tmp_path):
    cv2.setNumThreads(2)
    source = Path(__file__).resolve().parents[1] / 'videos' / 'mona-lisa.avi'
    cap = cv2.VideoCapture(str(source))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    clip = tmp_path / 'input.avi'
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    assert writer.isOpened()
    try:
        for _ in range(12):
            ok, frame = cap.read()
            assert ok
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    rows = []
    for mode in ('sift', 'hybrid'):
        out = tmp_path / 'mona-lisa' / mode
        result = process(clip, out, mode=mode)
        assert result['frames'] == 12
        assert result['evaluated'] == 11
        assert 0 <= result['coverage'] <= 1
        cap = cv2.VideoCapture(str(out / 'tracked.mp4'))
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(fps, abs=.02)
        cap.release()
        rows.append(dict(result, case='mona-lisa'))
    with (tmp_path / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = generate_report(tmp_path)
    assert 'mona-lisa' in report
    assert (tmp_path / 'coverage.png').exists()
    assert (tmp_path / 'fps.png').exists()
