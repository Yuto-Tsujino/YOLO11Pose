import cv2
import numpy as np
from config import GAUSSIAN_KERNEL_SIZE

def smooth_trajectory(trajectory, kernel_size=GAUSSIAN_KERNEL_SIZE):
    """軌跡をガウジアン平滑化する"""
    if len(trajectory) < kernel_size:
        return trajectory

    # x,y座標を分離
    x = np.array([p[0] for p in trajectory])
    y = np.array([p[1] for p in trajectory])

    # ガウシアンカーネルを作成して適用
    kernel = cv2.getGaussianKernel(kernel_size, -1)
    kernel = kernel.flatten()

    # 畳み込みを適用する
    pad = kernel_size // 2
    x_smooth = np.convolve(x, kernel, mode='same')
    y_smooth = np.convolve(y, kernel, mode='same')

    # 端の補正
    x_smooth[:pad] = x[:pad]
    x_smooth[-pad:] = x[-pad:]
    y_smooth[:pad] = y[:pad]
    y_smooth[-pad:] = y[-pad:]

    return list(zip(x_smooth.astype(int), y_smooth.astype(int)))
