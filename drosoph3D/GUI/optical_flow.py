import cv2
import numpy as np

def optical_flow(img_before, img_after, pts2d):
    lk_params = dict(winSize=(100, 100),
                     maxLevel=0,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
    pts2d = pts2d[:, np.newaxis, :]

    p1, st, err = cv2.calcOpticalFlowPyrLK(img_before, img_after, pts2d.astype(np.float32), None, **lk_params)
    return p1.reshape(-1, 2)

