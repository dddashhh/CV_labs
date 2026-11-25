import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lab2.tracker import valid_quad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    args = parser.parse_args()
    cap = cv2.VideoCapture(str(args.video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError('Cannot read video')
    scale = min(1.0, 1100/frame.shape[1], 750/frame.shape[0])
    view = cv2.resize(frame, None, fx=scale, fy=scale)
    points = []
    name = 'Select TL, TR, BR, BL | Enter: save | R: reset'

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(name, click)
    try:
        while True:
            image = view.copy()
            for i, p in enumerate(points):
                cv2.circle(image, p, 5, (0, 255, 0), -1)
                cv2.putText(image, str(i+1), p, cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            if len(points) >= 2:
                cv2.polylines(image, [np.int32(points)], len(points)==4, (0, 255, 0), 2)
            cv2.imshow(name, image)
            key = cv2.waitKey(25) & 255
            if key == ord('r'):
                points.clear()
            elif key == 27:
                break
            elif key in (10, 13) and len(points) == 4:
                q = np.float32(points)/scale
                if valid_quad(q, frame.shape):
                    output = args.video.with_suffix('.json')
                    output.write_text(json.dumps(q.reshape(-1).tolist()))
                    print(output)
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
