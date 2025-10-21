import numpy as np
import cv2
from ultralytics import YOLO
import torch
import config

#モデル読み込み
def load_model(model_path):
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    #GPU設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path).to(device) 
    #モデルを半精度で実行してメモリと処理速度を最適化
    model.half()
    return model

# ================== 姿勢推定と視線位置計算 ==================
def calculate_head_pose(left_eye, right_eye, nose):
    """頭の姿勢角（Pitch, Yaw）の計算"""
    le, re, n = np.array(left_eye), np.array(right_eye), np.array(nose)
    if np.all(le == 0) or np.all(re == 0) or np.all(n == 0):
        return 0.0, 0.0

    eyes_center = (le + re) / 2
    eye_dist = np.linalg.norm(le - re)  # 両目の中心

    # yaw（水平方向）を計算する
    sensitive_yaw = 50  # この値を調整することで、yawの感度が変わる。
    yaw = np.clip((n[0] - eyes_center[0]) / sensitive_yaw * 60, -60, 60)

    sensitive_pitch = 50  # この値を調整することで、pitchの感度が変わる
    pitch = np.clip((n[1] - eyes_center[1]) / sensitive_pitch * 60, -60, 60)
    pitch = -pitch
    return pitch, yaw

def calculate_screen_point(pitch_deg, yaw_deg, distance=1.0):
    """ベクトルによるスクリーン注視点の座標計算"""
    yaw_rad = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)

    # 方向ベクトルを計算する（YawはX軸に、PitchはY軸に影響する）
    vec_x = np.array([np.tan(yaw_rad), 0, 1])
    vec_y = np.array([0, np.tan(pitch_rad), 1])
    vec_dir = vec_x + vec_y
    vec_dir /= np.linalg.norm(vec_dir)  # 正規化
    # スクリーン（Z=0平面）との交点を計算します。
    t = distance / vec_dir[2]
    screen_x = t * vec_dir[0]
    screen_y = t * vec_dir[1]
    return screen_x, screen_y

def draw_joints_with_names(image, joints, keypoint_names, target_indices=[]):
    """キーポイントと名前のマッピング"""
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()

    for i in target_indices:
        if i < len(joints):
            x, y = int(joints[i][0]), int(joints[i][1])
            if x == 0 and y == 0:
                continue
            name = keypoint_names.get(i, f"unknown_{i}")
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image, f"{name}", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return image

def estimate_3d_point_from_calibration(px, py, fx, fy, cx, cy, Z_estimate, camera_matrix):
    """ピクセル座標(px, py)と仮のZ値から3D空間座標(X, Y, Z)を推定する
    -fx, fy: 焦点距離(ピクセル単位)
    -cx, cy: 光学中心(画像中心のx, y)
    -Z_estimate: 擬似的な奥行き(mm)
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    X = (px - cx) * Z_estimate / fx
    Y = (py - cy) * Z_estimate / fy
    return np.array([X, Y, Z_estimate])

def estimate_depth_from_2d_joints(joint1, joint2, real_distance_mm, camera_matriz):
    """
    2D関節座標2点と現実世界の距離から擬似Z値を推定する関数

    -joint1, joint2:(x, y)形式の2D関節座標(例:joints[5][:2])
    -real_distance_mm:現実世界での2転換距離(mm)
    -camera_matrix:カメラ内部パラメータ行列

    Returns:
    -推定Z値(mm)
    """
    ERROR_Z_VALUE = 1e7
    if joint1 is None or joint2 is None or np.all(joint1 == 0) or np.all(joint2 == 0):
        return ERROR_Z_VALUE #デフォルト値(エラー回避)
    
    pixel_dist = np.linalg.norm(np.array(joint1) - np.array(joint2))
    if pixel_dist == 0:
        return ERROR_Z_VALUE #デフォルト値(不明なとき)
    
    fx = camera_matriz[0, 0] 
    fy = camera_matriz[1, 1] 
    f = (fx + fy) / 2 # 焦点距離
    Z_estimate = f * real_distance_mm / pixel_dist
    return Z_estimate

def get_all_joints_3d_coordinates(
        joints_2d_map_indexed,      #例:{"LEFT_SHOULDER":[x, y], "LEFT_ELBOW":[x, y]}
        bone_definitions_named,   #例:[("LEFT_SHOULDER", "LEFT_ELBOW", "UPPER_ARM_LENGTH"),...]
        real_lengths_map,   #例:{"UPPER_ARM_LENGTH":300, ...}
        camera_matrix,
        kp_names_map        #例:config.KP_NAMES
):
    """
    全ての関節の3D座標を推定する関数。

    Args:
        joints_2d_map (dict): 関節名をキー、2D座標 (x,y)またはNoneを値とする辞書。
        bone_definitions (list): (関節名1, 関節名2, 長さ名) のタプルリスト。
                                 関節名1と関節名2で骨セグメントを定義し、その長さを長さ名で参照。
        real_lengths_map (dict): 長さ名をキー、実際の長さ(mm)を値とする辞書。
        camera_matrix (np.ndarray): カメラ内部パラメータ行列。

    Returns:
        tuple: (
            joints_3d_map (dict): 関節名をキー、3D座標 [X,Y,Z]またはNoneを値とする辞書,
            joint_final_Z_values (dict): 関節名をキー、推定されたZ値またはエラー値を値とする辞書
        )
    """
    ERROR_Z_THRESHOLD = 1e7 - 1 # これより大きいZ値はエラーとみなす閾値

    #関数名をインデックスに、インデックスを関節名に変換する辞書を作成
    idx_to_name = kp_names_map
    name_to_idx = {v: k for k, v in idx_to_name.items()}

    # camera_matrixからfx, fy, cx, cyを取得
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # 1. 各セグメントのZ値を計算し、各関節にZ値の候補を紐付ける
    joint_Z_candidates_indexed = {} # { joint_idx: [Z候補1, Z候補2, ...], ... }

    for joint1_name, joint2_name, length_name in bone_definitions_named:
        joint1_idx = name_to_idx.get(joint1_name)
        joint2_idx = name_to_idx.get(joint2_name)

        if joint1_idx is None or joint2_idx is None:
            continue
        joint1_2d = joints_2d_map_indexed.get(joint1_idx)
        joint2_2d = joints_2d_map_indexed.get(joint2_idx)
        real_dist_mm = real_lengths_map.get(length_name)

        if joint1_2d is not None and joint2_2d is not None and real_dist_mm is not None:
            # 座標が(0,0)でないことを確認
            if np.all(np.array(joint1_2d) == 0) or np.all(np.array(joint2_2d) == 0):
                segment_Z = 1e7 # 無効な座標の場合はエラーZ
            else:
                segment_Z = estimate_depth_from_2d_joints(joint1_2d, joint2_2d, real_dist_mm, camera_matrix)        
            # このセグメントZを、構成する両方の関節のZ値候補として追加
            if segment_Z <= ERROR_Z_THRESHOLD: # 有効なZ値のみ追加
                for joint_idx in [joint1_idx, joint2_idx]:
                    if joint_idx not in joint_Z_candidates_indexed:
                        joint_Z_candidates_indexed[joint_idx] = []
                    joint_Z_candidates_indexed[joint_idx].append(segment_Z)
    
    # 2. 各関節の最終的なZ値を決定
    joint_final_Z_values_indexed = {} # { joint_idx: 最終Z値, ... }
    # bone_definitions_named に含まれるすべての関節インデックスを取得
    all_involved_joint_indices = set()
    for j1_name, j2_name, _ in bone_definitions_named:
        idx1 = name_to_idx.get(j1_name)
        idx2 = name_to_idx.get(j2_name)
        if idx1 is not None: all_involved_joint_indices.add(idx1)
        if idx2 is not None: all_involved_joint_indices.add(idx2)

    for joint_idx in all_involved_joint_indices:
        if joint_idx in joint_Z_candidates_indexed and joint_Z_candidates_indexed[joint_idx]:
            valid_Zs = [z for z in joint_Z_candidates_indexed[joint_idx] if z <= ERROR_Z_THRESHOLD]
            if valid_Zs:
                joint_final_Z_values_indexed[joint_idx] = np.mean(valid_Zs)
            else:
                joint_final_Z_values_indexed[joint_idx] = 1e7 
        else:
            joint_final_Z_values_indexed[joint_idx] = 1e7

    # 3. 各関節の2D座標と決定されたZ値から3D座標を計算
    joints_3d_map_indexed = {} # { joint_idx: [X,Y,Z] or None, ... }
    for joint_idx, joint_2d_coords in joints_2d_map_indexed.items(): # 元の2Dキーポイントすべてに対して処理
        if joint_2d_coords is None or np.all(np.array(joint_2d_coords) == 0):
            joints_3d_map_indexed[joint_idx] = None
            if joint_idx not in joint_final_Z_values_indexed:
                joint_final_Z_values_indexed[joint_idx] = 1e7
            continue

        px, py = joint_2d_coords
        estimated_Z = joint_final_Z_values_indexed.get(joint_idx, 1e7) 

        if estimated_Z <= ERROR_Z_THRESHOLD:
            joints_3d_map_indexed[joint_idx] = estimate_3d_point_from_calibration(
                px, 
                py, 
                fx,
                fy,
                cx,
                cy,
                estimated_Z, 
                camera_matrix
            )
        else:
            joints_3d_map_indexed[joint_idx] = None
            # joint_final_Z_values_indexed にはエラー値がそのまま入っている
    # joints_3d_map_indexed と joint_final_Z_values_indexed のキーが kp_names_map の全キーを網羅するようにする
    for kp_idx in kp_names_map.keys():
        if kp_idx not in joints_3d_map_indexed:
            joints_3d_map_indexed[kp_idx] = None
        if kp_idx not in joint_final_Z_values_indexed:
            joint_final_Z_values_indexed[kp_idx] = 1e7

    return joints_3d_map_indexed, joint_final_Z_values_indexed

def scale_3d_point(p1, p2, real_distance):
    """
    2点間の実距離を基準として、3D点をスケーリングする
    -p1, p2: 3D座標のnp.array
    -real_distance: 現実世界での距離(メートル)
    """

    measured_length = np.linalg.norm(p2 - p1)
    if measured_length == 0:
        return p1, p2
    scale = real_distance / measured_length
    return p1 * scale, p2 * scale
