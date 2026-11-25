from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Estimate:
    corners: np.ndarray
    reference: np.ndarray
    points: np.ndarray
    inliers: int
    ratio: float
    error: float
    method: str


def frame_corners(shape):
    h, w = shape[:2]
    return np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])


def valid_quad(q, shape):
    h, w = shape[:2]
    q = np.asarray(q, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(q).all() or not cv2.isContourConvex(q):
        return False
    area = cv2.contourArea(q, oriented=True)
    if not 100 <= area <= 4 * h * w:
        return False
    if np.min(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)) < 5:
        return False
    visible, _ = cv2.intersectConvexConvex(q, frame_corners(shape))
    return visible / area >= 0.35


def polygon_iou(a, b):
    if a is None or b is None:
        return 0.0
    a, b = np.float32(a).reshape(4, 2), np.float32(b).reshape(4, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


def estimate_homography(reference, points, corners, shape, method, threshold=3.0, min_inliers=8, min_ratio=0.4):
    reference = np.float32(reference).reshape(-1, 2)
    points = np.float32(points).reshape(-1, 2)
    if len(reference) < min_inliers or len(reference) != len(points):
        return None
    h, mask = cv2.findHomography(reference, points, cv2.RANSAC, threshold, maxIters=2000, confidence=0.995)
    if h is None or mask is None or not np.isfinite(h).all():
        return None
    good = mask.ravel().astype(bool)
    count = int(good.sum())
    ratio = count / len(good)
    if count < min_inliers or ratio < min_ratio:
        return None
    src, dst = reference[good], points[good]
    ref_area = cv2.contourArea(np.float32(corners))
    if cv2.contourArea(cv2.convexHull(src)) < 0.025 * ref_area:
        return None
    denominators = np.c_[corners, np.ones(4)] @ h[2]
    if np.any(np.abs(denominators) < 1e-8) or not (np.all(denominators > 0) or np.all(denominators < 0)):
        return None
    q = cv2.perspectiveTransform(np.float32(corners)[None], h)[0]
    if not valid_quad(q, shape):
        return None
    projected = cv2.perspectiveTransform(src[None], h)[0]
    error = float(np.sqrt(np.mean(np.sum((projected - dst) ** 2, axis=1))))
    return Estimate(q, src, dst, count, ratio, error, method)


class Tracker:
    def __init__(self, frame, corners=None, mode='hybrid', interval=10):
        self.mode = mode
        self.interval = interval
        self.index = 0
        self.detector = cv2.SIFT_create(nfeatures=1600, contrastThreshold=0.025)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.corners = frame_corners(frame.shape) if corners is None else np.float32(corners).reshape(4, 2)
        if not valid_quad(self.corners, frame.shape):
            raise ValueError('Invalid reference quadrilateral')
        gray = self.gray(frame)
        self.ref_points, self.descriptors = self.reference_features(gray)
        if len(self.ref_points) < 8:
            raise ValueError('Reference contains too few keypoints')
        self.previous_gray = gray
        self.previous = Estimate(self.corners.copy(), self.ref_points, self.ref_points.copy(), len(self.ref_points), 1.0, 0.0, 'reference')
        self.reseed(gray, self.previous)

    def gray(self, frame):
        return self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    def reference_features(self, gray):
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.round(self.corners).astype(np.int32), 255)
        coordinates, descriptors = [], []
        for scale in (1.0, 0.65, 0.4, 0.25):
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            points, desc = self.detector.detectAndCompute(small, small_mask)
            if desc is not None:
                coordinates.extend((p.pt[0] / scale, p.pt[1] / scale) for p in points)
                descriptors.append(desc)
        return np.float32(coordinates).reshape(-1, 2), np.vstack(descriptors) if descriptors else None

    def detect(self, gray):
        points, desc = self.detector.detectAndCompute(gray, None)
        if desc is None or len(desc) < 2:
            return None
        pairs = self.matcher.knnMatch(self.descriptors, desc, k=2)
        candidates = sorted((pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance), key=lambda m: m.distance)
        used = set()
        matches = []
        for m in candidates:
            if m.trainIdx not in used:
                matches.append(m)
                used.add(m.trainIdx)
        return estimate_homography([self.ref_points[m.queryIdx] for m in matches], [points[m.trainIdx].pt for m in matches], self.corners, gray.shape, 'sift')

    def flow(self, gray):
        if self.previous is None or len(self.previous.points) < 8:
            return None
        p = self.previous.points.reshape(-1, 1, 2).astype(np.float32)
        options = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        q, s, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, p, None, **options)
        if q is None:
            return None
        back, b, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, q, None, **options)
        if back is None:
            return None
        good = (s.ravel() == 1) & (b.ravel() == 1) & (np.linalg.norm((p - back).reshape(-1, 2), axis=1) < 1.5)
        good &= np.isfinite(q.reshape(-1, 2)).all(axis=1)
        if good.sum() < 8 or good.mean() < 0.35:
            return None
        return estimate_homography(self.previous.reference[good], q.reshape(-1, 2)[good], self.corners, gray.shape, 'lk', threshold=2.5, min_ratio=0.6)

    def reseed(self, gray, estimate):
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.round(estimate.corners).astype(np.int32), 255)
        mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
        points = cv2.goodFeaturesToTrack(gray, maxCorners=220, qualityLevel=0.01, minDistance=5, mask=mask)
        if points is None or len(points) < 8:
            return
        inverse = cv2.getPerspectiveTransform(estimate.corners.astype(np.float32), self.corners.astype(np.float32))
        ref = cv2.perspectiveTransform(points, inverse).reshape(-1, 2)
        if np.isfinite(ref).all():
            estimate.reference = ref
            estimate.points = points.reshape(-1, 2)

    def update(self, frame):
        self.index += 1
        gray = self.gray(frame)
        flow = self.flow(gray) if self.mode == 'hybrid' else None
        detection = self.detect(gray) if self.mode == 'sift' or flow is None or self.index % self.interval == 0 or flow.inliers < 15 else None
        if detection is not None and flow is not None and polygon_iou(detection.corners, flow.corners) < 0.60 and detection.inliers < 25:
            detection = None
        result = detection if detection is not None else flow
        if result is not None and (result.method == 'sift' or len(result.points) < 45):
            self.reseed(gray, result)
        self.previous_gray = gray
        self.previous = result
        return result


def draw(frame, estimate, label):
    output = frame.copy()
    if estimate is None:
        cv2.putText(output, f'{label}: not found', (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 40, 240), 2, cv2.LINE_AA)
        return output
    q = np.round(estimate.corners).astype(np.int32)
    cv2.polylines(output, [q], True, (50, 230, 70), 2, cv2.LINE_AA)
    x = max(5, min(int(q[:, 0].min()), frame.shape[1] - 180))
    y = max(22, min(int(q[:, 1].min()) - 8, frame.shape[0] - 8))
    cv2.putText(output, f'{label} | {estimate.method}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 230, 70), 2, cv2.LINE_AA)
    return output
