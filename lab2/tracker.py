"""SIFT для обнаружения, оптический поток для слежения между кадрами."""

import cv2
import numpy as np


class PlanarTracker:
    def __init__(self, frame, roi=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        x, y, w, h = roi if roi is not None else (0, 0, width, height)
        if x < 0 or y < 0 or w < 2 or h < 2 or x + w > width or y + h > height:
            raise ValueError("Область объекта должна находиться внутри первого кадра")
        mask = np.zeros_like(gray)
        mask[y:y + h, x:x + w] = 255
        self.sift = cv2.SIFT_create(nfeatures=1500)
        self.reference_points, self.descriptors = self.sift.detectAndCompute(gray, mask)
        if self.descriptors is None or len(self.reference_points) < 15:
            raise ValueError("На первом кадре слишком мало ключевых точек")
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.corners = np.float32([[x, y], [x, y + h - 1],
                                  [x + w - 1, y + h - 1], [x + w - 1, y]]).reshape(-1, 1, 2)
        self.matrix = np.eye(3)
        self.previous = gray
        self.points = None
        self.anchors = None
        self.misses = 0
        self.seed(gray)

    def polygon(self, matrix):
        if matrix is None or not np.isfinite(matrix).all():
            return None
        polygon = cv2.perspectiveTransform(self.corners, matrix)
        if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):
            return None
        height, width = self.previous.shape
        if not 400 < cv2.contourArea(polygon) < 1.5 * width * height:
            return None
        vertices = polygon.reshape(-1, 2)
        sides = np.linalg.norm(vertices - np.roll(vertices, 1, axis=0), axis=1)
        if sides.min() < 1 or sides.max() / sides.min() > 6:
            return None
        if (vertices[:, 0].min() < -width or vertices[:, 0].max() > 2 * width
                or vertices[:, 1].min() < -height or vertices[:, 1].max() > 2 * height):
            return None
        return polygon

    def detect(self, gray):
        points, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(points) < 15:
            return None
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = [pair[0] for pair in pairs
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
        if len(good) < 15:
            return None
        source = np.float32([self.reference_points[m.queryIdx].pt for m in good])
        target = np.float32([points[m.trainIdx].pt for m in good])
        matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if mask is None or mask.sum() < 15 or self.polygon(matrix) is None:
            return None
        return matrix

    def seed(self, gray):
        # Координаты в шаблоне сохраняются, чтобы не перемножать гомографии.
        self.points = self.anchors = None
        polygon = self.polygon(self.matrix)
        if polygon is None:
            return
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.int32(polygon), 255)
        points = cv2.goodFeaturesToTrack(gray, 200, 0.001, 5, mask=mask)
        if points is not None:
            try:
                inverse = np.linalg.inv(self.matrix)
            except np.linalg.LinAlgError:
                return
            anchors = cv2.perspectiveTransform(points, inverse)
            if np.isfinite(anchors).all():
                self.points, self.anchors = points, anchors

    def follow(self, gray):
        if self.points is None or len(self.points) < 8:
            return None
        options = dict(winSize=(21, 21), maxLevel=4,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous, gray, self.points, None, **options)
        if moved is None:
            return None
        returned, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous, moved, None, **options)
        if returned is None:
            return None
        error = np.linalg.norm(returned - self.points, axis=2).ravel()
        valid = (status.ravel() == 1) & (back_status.ravel() == 1) & (error < 2.0)
        if valid.sum() < 8:
            return None
        source, target = self.anchors[valid], moved[valid]
        matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if mask is None or mask.sum() < 8 or self.polygon(matrix) is None:
            return None
        keep = mask.ravel() == 1
        self.points, self.anchors = target[keep], source[keep]
        return matrix

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matrix = self.detect(gray)
        mode = "detect"
        if matrix is None and self.matrix is not None:
            matrix = self.follow(gray)
            mode = "flow"
        if matrix is not None:
            self.matrix = matrix
            self.misses = 0
            if mode == "detect" or self.points is None or len(self.points) < 100:
                self.seed(gray)
        elif self.matrix is not None and self.misses < 5:
            # Короткий смаз: рамку держим, точки ищем заново на текущем кадре.
            self.misses += 1
            mode = "hold"
            self.seed(gray)
        else:
            self.matrix = None
            self.points = self.anchors = None
            mode = "lost"
        self.previous = gray
        return self.polygon(self.matrix), mode
