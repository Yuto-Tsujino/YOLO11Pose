import numpy as np
import cv2


# カメラに写っているぶぶんのみ線を引く
def safe_draw_line(img, pt1, pt2, color, thickness=2):
    if np.all(pt1 == np.array([0,0])) or np.all(pt2 == np.array([0,0])):
        return
    else:
        cv2.line(img, tuple(pt1.astype(int)), tuple(pt2.astype(int)), color, thickness)
