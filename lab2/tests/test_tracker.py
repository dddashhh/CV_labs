import cv2
import numpy as np
import pytest

from lab2.tracker import Tracker, draw, estimate_homography, frame_corners, polygon_iou, valid_quad


def texture():
    rng = np.random.default_rng(19)
    image = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def test_homography_with_outliers():
    rng = np.random.default_rng(42)
    src = rng.uniform([0, 0], [319, 239], (70, 2)).astype(np.float32)
    corners = frame_corners((240, 320))
    expected = np.float32([[30, 35], [270, 15], [290, 200], [40, 220]])
    h = cv2.getPerspectiveTransform(corners, expected)
    dst = cv2.perspectiveTransform(src[None], h)[0]
    dst[:15] = rng.uniform([0, 0], [319, 239], (15, 2))
    result = estimate_homography(src, dst, corners, (240, 320), 'sift')
    assert result is not None
    assert result.inliers == 55
    assert polygon_iou(result.corners, expected) > .999


def test_invalid_geometry():
    q = frame_corners((240, 320))
    assert valid_quad(q, (240, 320))
    assert not valid_quad(q[[0, 2, 1, 3]], (240, 320))
    assert not valid_quad(q[::-1], (240, 320))
    assert not valid_quad(q + 1000, (240, 320))
    assert not valid_quad(q * .001, (240, 320))


def test_collinear_points():
    points = np.float32([[i, i] for i in range(20)])
    assert estimate_homography(points, points, frame_corners((240, 320)), (240, 320), 'sift') is None


def test_iou():
    a = np.float32([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert polygon_iou(a, a) == pytest.approx(1)
    assert polygon_iou(a, a + [5, 0]) == pytest.approx(1/3)
    assert polygon_iou(a, None) == 0


@pytest.mark.parametrize('mode', ['sift', 'hybrid'])
def test_translation_and_recovery(mode):
    cv2.setNumThreads(2)
    first = texture()
    tracker = Tracker(first, mode=mode)
    h = np.float32([[.85, 0, 20], [0, .85, 15]])
    moved = cv2.warpAffine(first, h, (320, 240))
    q = cv2.transform(frame_corners(first.shape)[None], h)[0]
    result = tracker.update(moved)
    assert result is not None
    assert polygon_iou(result.corners, q) > .9
    assert tracker.update(np.zeros_like(first)) is None
    result = tracker.update(moved)
    assert result is not None
    assert polygon_iou(result.corners, q) > .9


def test_blank_reference():
    with pytest.raises(ValueError):
        Tracker(np.zeros((200, 300, 3), np.uint8))


def test_draw_does_not_change_input():
    image = texture()
    saved = image.copy()
    rendered = draw(image, None, 'Object')
    assert np.array_equal(image, saved)
    assert not np.array_equal(image, rendered)
