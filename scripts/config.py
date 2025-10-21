import numpy as np
import os

# XIMEA設定
XIMEA_EXPOSURE = 30000
XIMEA_GAIN = 0.0
XIMEA_AUTO_WB = 1 # 0:off, 1:on

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_PATH = os.path.join(BASE_DIR, "..", "..", "calibration_data.npz")

#モデルと距離設定
MODEL_PATH = "/home/zejing/posedetect/datasets/working/yolo11n-pose.pt"  # YOLO姿勢推定モデルのパス
SCREEN_DISTANCE = 1.0  # 顔からスクリーンまでの距離（メートル）

#キーポイント名とインデックス
KP_NAMES = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
}
NOSE_INDEX = 0
LEFT_EYE_INDEX = 1
RIGHT_EYE_INDEX = 2
LEFT_EAR_INDEX = 3
RIGHT_EAR_INDEX = 4
LEFT_SHOULDER_INDEX = 5
RIGHT_SHOULDER_INDEX = 6
LEFT_ELBOW_INDEX = 7
RIGHT_ELBOW_INDEX = 8
LEFT_WRIST_INDEX = 9
RIGHT_WRIST_INDEX = 10
LEFT_HIP_INDEX = 11
RIGHT_HIP_INDEX = 12
LEFT_KNEE_INDEX = 13
RIGHT_KNEE_INDEX = 14
LEFT_ANKLE_INDEX = 15
RIGHT_ANKLE_INDEX = 16

# 実際の骨の長さ
bone_real_lengths = {
    "upper_arm_len":300, #上腕長30cm
    "forearm_len":250, #前腕長25cm
    "shoulder_width":400, #肩幅40cm
    "body_length":800, #胴80cm
    "hip_length":300 #腰幅30cm
}

# 骨セグメントの定義(関節名1, 関節名2, その骨の名前)
body_bones = [
    ("left_shoulder", "left_elbow", "upper_arm_len"),
    ("left_elbow", "left_wrist", "forearm_len"),
    ("right_shoulder", "right_elbow", "upper_arm_len"),
    ("right_elbow", "right_wrist", "forearm_len"),
    ("left_shoulder", "right_shoulder", "shoulder_width"),
    # 体幹 (より詳細な定義も可能。ここでは肩と腰)
    ("left_shoulder", "left_hip", "body_length"),
    ("right_shoulder", "right_hip", "body_length"),
    # 腰
    ("left_hip", "right_hip", "hip_length"),
]

#色定義(BGR)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)  # ← この行を追加または確認
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BLUE = (255, 191, 0) # 例
PURPLE = (128, 0, 128)    # 例

#アート風表示のパラメータ
TRAIL_LENGTH = 100  # 追跡軌跡の最大保存数
GAUSSIAN_KERNEL_SIZE = 15  # ガウジアンカーネルのサイズ
LINE_WIDTH = 3  # 線の基本太さ
MAX_ADDITIONAL_WIDTH = 5  # 線の追加太さ最大値
COLOR_GRADIENT = [(0, 255, 255), (0, 180, 255), (0, 100, 255)]  # 顔の色グラデーション



# カメラキャリブレーションデータの読み込み
try:
    calibration_data = np.load(os.path.normpath(CALIB_PATH))
    camera_matrix = calibration_data["camera_matrix"]
    dist_coeffs = calibration_data["dist_coeffs"]
except FileNotFoundError:
    print("[config.py] calibration_data.npzが見つかりません")
    camera_matrix = None
    dist_coeffs = None