import cv2
from trajectory import smooth_trajectory
from config import LINE_WIDTH, MAX_ADDITIONAL_WIDTH, COLOR_GRADIENT

def draw_artistic_trail(image, trajectory):
    """軌跡にアート風エフェクトを適用して描画する"""
    if len(trajectory) < 2:
        return image

    # ガウス・スムージングの適用
    smoothed_trail = smooth_trajectory(trajectory, config.GAUSSIAN_KERNEL_SIZE)

    # グラデーションの幅と色で線を描く
    for i in range(1, len(smoothed_trail)):
        # 軌跡における現在点の位置の割合を計算する（0-1)
        pos_ratio = i / len(smoothed_trail)

        # 線幅を計算する - 点が新しいほど線幅は大きくなります。 
        width = config.LINE_WIDTH + int(config.MAX_ADDITIONAL_WIDTH * (1 - pos_ratio))

        # グラデーションを使って色を計算する
        color_idx = min(int(pos_ratio * len(config.COLOR_GRADIENT)), len(config.COLOR_GRADIENT) - 1)
        color = config.COLOR_GRADIENT[color_idx]

        # 線を引く
        cv2.line(image, smoothed_trail[i - 1], smoothed_trail[i], color, width)

    # 最新点にハロー効果を加える
    last_point = smoothed_trail[-1]
    cv2.circle(image, last_point, width * 2, (255, 255, 255), 1)
    cv2.circle(image, last_point, width, color, -1)

    return image
