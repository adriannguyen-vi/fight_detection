from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import numpy as np
import collections


def iou_batch(bb_test, bb_gt):
    bb_test = np.expand_dims(bb_test, 1)
    bb_gt   = np.expand_dims(bb_gt, 0)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)

    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) *
        (bb_test[..., 3] - bb_test[..., 1]) +
        (bb_gt[..., 2] - bb_gt[..., 0]) *
        (bb_gt[..., 3] - bb_gt[..., 1]) -
        wh
    )
    return o


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox, cls=None):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        self.kf.H = np.eye(4, 7)

        self.kf.x[:4] = bbox.reshape((4, 1))
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.hits = 0
        self.no_losses = 0

        self.cls = cls
    def predict(self):
        self.kf.predict()
        return self.kf.x[:4].reshape(-1)

    def update(self, bbox, cls = None):
        self.kf.update(bbox.reshape((4, 1)))
        self.hits += 1
        self.no_losses = 0
        if cls is not None:
            self.cls = cls

class Sort:
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.trackers = []
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def update(self, dets):
        """
        dets: Nx6 -> [x1,y1,x2,y2,score,cls]
        """
        if len(self.trackers) == 0:
            for d in dets:
                self.trackers.append(
                    KalmanBoxTracker(d[:4], int(d[5]))
                )
        else:
            trks = np.array([t.predict() for t in self.trackers])

            iou_mat = iou_batch(trks, dets[:, :4]) if len(dets) else np.empty((0, 0))
            matched_idx = linear_sum_assignment(-iou_mat) if iou_mat.size else ([], [])

            matched_dets = set()

            for t_idx, d_idx in zip(*matched_idx):
                if iou_mat[t_idx, d_idx] >= self.iou_threshold:
                    self.trackers[t_idx].update(
                        dets[d_idx, :4],
                        int(dets[d_idx, 5])
                    )
                    matched_dets.add(d_idx)

            # Create new trackers for unmatched detections
            for i, d in enumerate(dets):
                if i not in matched_dets:
                    self.trackers.append(
                        KalmanBoxTracker(d[:4], int(d[5]))
                    )

        # Age & output
        ret = []
        for t in self.trackers[:]:
            t.no_losses += 1
            if t.no_losses > self.max_age:
                self.trackers.remove(t)
                continue

            bbox = t.kf.x[:4].reshape(-1)
            ret.append([*bbox, t.id, t.cls])

        return np.array(ret)
    
    def reset(self):
        self.trackers.clear()
        KalmanBoxTracker.count = 0


