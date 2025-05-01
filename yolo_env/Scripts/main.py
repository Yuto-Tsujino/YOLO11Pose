# from ultralytics import YOLO

# model = YOLO("yolo11n.pt")  # モデルのパスを指定

# # 画像や動画の処理
# results = model("bus.jpg")  # 画像を推論
# results[0].show()  # 結果を表示

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
from ximea import xiapi
import threading
import queue

# ================== パラメータ設定 ==================
MODEL_PATH = "/home/zejing/posedetect/datasets/working/yolo11n-pose.pt"  # YOLOv8姿勢推定モデルのパス
SCREEN_DISTANCE = 1.0  # 顔からスクリーンまでの距離（メートル）

#GPU設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(MODEL_PATH).to(device)
#モデルを半精度で実行してメモリと処理速度を最適化
model.half()

KP_NAMES = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "righqt_elbow",
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

# 色
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
LIGHT_BLUE = (255, 255, 0)


# アート風表示のパラメータ
TRAIL_LENGTH = 100  # 追跡軌跡の最大保存数
GAUSSIAN_KERNEL_SIZE = 15  # ガウジアンカーネルのサイズ
LINE_WIDTH = 3  # 線の基本太さ
MAX_ADDITIONAL_WIDTH = 5  # 線の追加太さ最大値
COLOR_GRADIENT = [(0, 255, 255), (0, 180, 255), (0, 100, 255)]  # 顔の色グラデーション

# キューを定義
frame_queue = queue.Queue(maxsize = 1)

# ================== 補助関数 ==================
def camere_thread_func(cap, image):
    while True:
        try:
            cap.get_image(image)
            frame = image.get_image_data_numpy()
            if frame is not None and frame.size > 0:
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                #古いフレームを破棄
                if not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except:
                        pass

                frame_queue.put_nowait(frame)
        except Exception as e:
            time.sleep(1)
            print(f"Camera error: {e}")
            continue
#解像度を落としたかっただけなのに
# def resize_and_adjust_keypoints(frame, keypoints, target_size = (320, 240)):
#     # 元の画像サイズ
#     original_height, original_width  = frame.shape[:2]

#     # 画像のリサイズ
#     frame_resized = cv2.resize(frame, target_size)

#     # 新しい画像のサイズ
#     new_height, new_width = frame_resized[:2]

#     # スケーリング係数
#     scale_x = new_width / original_width
#     scale_y = new_height / original_height

#     keypoints_resized = keypoints * [scale_x, scale_y]

#     return frame_resized, keypoints_resized

# def process_frame(frame):
#     # YOLO推論
#     results = model(frame)
#     keypoints = results.keypoints

#     if len(results > 0):
#         result = results[0]
#         keypoints = result.keypoints

#         # キーポイントが検出された場合
#         if keypoints is not None and len(keypoints) > 0:
#             keypoints = keypoints[0].cpu().numpy()  # 最初の結果を取得

#             # リサイズしたフレームとキーポイント
#             frame_resized, keypoints_resized = resize_and_adjust_keypoints(frame, keypoints)

#             # キーポイントを描画
#             frame_resized = draw_joints_with_names(frame_resized, keypoints_resized, KP_NAMES)

#             # 目の位置から姿勢を計算
#             if keypoints_resized[LEFT_EYE_INDEX][0] != 0 and keypoints_resized[RIGHT_EYE_INDEX][0] != 0 and keypoints_resized[NOSE_INDEX][0] != 0:
#                 pitch, yaw = calculate_head_pose(keypoints_resized[LEFT_EYE_INDEX], keypoints_resized[RIGHT_EYE_INDEX], keypoints_resized[NOSE_INDEX])

#                 # 頭の姿勢を表示
#                 print(f"Pitch: {pitch}, Yaw: {yaw}")

#                 # スクリーンの注視点を計算
#                 screen_x, screen_y = calculate_screen_point(pitch, yaw, SCREEN_DISTANCE)

#                 # スクリーン上の注視点を描画
#                 cv2.circle(frame_resized, (int(screen_x * 100 + 160), int(screen_y * 100 + 120)), 5, (0, 255, 255), -1)

#             # アート風の軌跡を描画
#             # ここでは追跡するキーポイントのペアを設定（例：目、鼻、肩など）
#             target_points = [LEFT_EYE_INDEX, RIGHT_EYE_INDEX, NOSE_INDEX]
#             trajectory = [(keypoints_resized[i][0], keypoints_resized[i][1]) for i in target_points]
#             frame_resized = draw_artistic_trail(frame_resized, trajectory)

#             return frame_resized, results

#     return frame, results

def smooth_trajectory(trajectory, kernel_size=5):
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


def draw_artistic_trail(image, trajectory):
    """軌跡にアート風エフェクトを適用して描画する"""
    if len(trajectory) < 2:
        return image

    # ガウス・スムージングの適用
    smoothed_trail = smooth_trajectory(trajectory, GAUSSIAN_KERNEL_SIZE)

    # グラデーションの幅と色で線を描く
    for i in range(1, len(smoothed_trail)):
        # 軌跡における現在点の位置の割合を計算する（0-1)
        pos_ratio = i / len(smoothed_trail)

        # 線幅を計算する - 点が新しいほど線幅は大きくなります。 
        width = LINE_WIDTH + int(MAX_ADDITIONAL_WIDTH * (1 - pos_ratio))

        # グラデーションを使って色を計算する
        color_idx = min(int(pos_ratio * len(COLOR_GRADIENT)), len(COLOR_GRADIENT) - 1)
        color = COLOR_GRADIENT[color_idx]

        # 線を引く
        cv2.line(image, smoothed_trail[i - 1], smoothed_trail[i], color, width)

    # 最新点にハロー効果を加える
    last_point = smoothed_trail[-1]
    cv2.circle(image, last_point, width * 2, (255, 255, 255), 1)
    cv2.circle(image, last_point, width, color, -1)

    return image

# カメラに写っているぶぶんのみ線を引く
def safe_draw_line(img, pt1, pt2, color, thickness=2):
    if np.all(pt1 == np.array([0,0])) or np.all(pt2 == np.array([0,0])):
        return
    else:
        cv2.line(img, tuple(pt1.astype(int)), tuple(pt2.astype(int)), color, thickness)


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


# =================メイン処理 ==================
def main():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    cap = xiapi.Camera()

    #FPS計算のための変数
    WARMUP_FRAME = 10
    frame_count = 0
    total_time = 0
    start_measurement = False

    torch.cuda.is_available()
    x = 1

    try:
        cap.open_device()
        print("Camera successfully opened.")
        cap.start_acquisition()  # Acquisition（画像取得）を開始
        # 露出時間（マイクロ秒）→ 明るくするには大きく
        cap.set_exposure(int(30000))  # 例：20000us = 20ms（デフォルトより明るめ）
        # ゲイン（増幅）→ ノイズも増えるけど暗さ対策になる
        cap.set_param('gain', 0.0)  # デフォルトは0.0、範囲は0〜最大値（カメラによる）
        # ホワイトバランスを自動にする（カラー画像が正しく見えるように）
        cap.set_param('auto_wb', 1)
        print("Camera acquisition started.")
    except Exception as e:
        print(f"Error: {e}")

    trajectory = []
    image = xiapi.Image()
    time.sleep(1)  # 少し待機してから画像を取得
    
    threading.Thread(target = camere_thread_func, args = (cap, image), daemon = True).start()
    while True:
        start_time = time.time()
        frame = None

        try:
            frame = frame_queue.get(timeout = 1) # 待機して取得
        except queue.Empty:
            print("Frame queue timeout.")
            continue

        imgsz = 192
        results = model(frame, imgsz)
        # while retry_count > 0:
        #     cap.get_image(image)
        #     frame = image.get_image_data_numpy()
        #     # カラー画像に変換（2次元→3次元）
        #     if len(frame.shape) == 2:
        #         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        #     if frame is not None:
        #         results = process_frame(frame)
        #         if frame.size > 0:
        #             print(f"Frame captured: {frame.shape}")
        #             break
        #     else:
        #         retry_count -= 1
        #         print(f"Failed to capture frame, {retry_count} retries left.")
        #         time.sleep(0.1)

        # if frame is None or frame.size == 0:
        #     print("Error: Failed to capture frame after multiple attempts.")
        #     break

        # print(f"Frame captured: {frame.shape}")  # 画像が取得できた場合に表示


        frame_pose = frame.copy()

        for result in results:
            if result.keypoints is None:
                continue

            kpts = result.keypoints.xy[0].cpu().numpy()  # CPUに戻して処理
            
            #for i, (x, y) in enumerate(kpts):
            #    print(f"{KP_NAMES.get(i, f'keypoint_{i}')}: ({x:.1f}, {y:.1f})")


            nose = kpts[NOSE_INDEX] if NOSE_INDEX < len(kpts) else np.array([0, 0])
            left_eye = kpts[LEFT_EYE_INDEX] if LEFT_EYE_INDEX < len(kpts) else np.array([0, 0])
            left_shoulder = kpts[LEFT_SHOULDER_INDEX] if LEFT_SHOULDER_INDEX < len(kpts) else np.array([0, 0])
            left_elbow = kpts[LEFT_ELBOW_INDEX] if LEFT_ELBOW_INDEX < len(kpts) else np.array([0, 0])
            left_wrist = kpts[LEFT_WRIST_INDEX] if LEFT_WRIST_INDEX < len(kpts) else np.array([0, 0])
            left_hip = kpts[LEFT_HIP_INDEX] if LEFT_HIP_INDEX < len(kpts) else np.array([0, 0])
            right_eye = kpts[RIGHT_EYE_INDEX] if RIGHT_EYE_INDEX < len(kpts) else np.array([0, 0])
            right_shoulder = kpts[RIGHT_SHOULDER_INDEX] if RIGHT_SHOULDER_INDEX < len(kpts) else np.array([0, 0])
            right_elbow = kpts[RIGHT_ELBOW_INDEX] if RIGHT_ELBOW_INDEX < len(kpts) else np.array([0, 0])
            right_wrist = kpts[RIGHT_WRIST_INDEX] if RIGHT_WRIST_INDEX < len(kpts) else np.array([0, 0])
            right_hip = kpts[RIGHT_HIP_INDEX] if RIGHT_HIP_INDEX < len(kpts) else np.array([0, 0])
            screen_pos = kpts[RIGHT_HIP_INDEX] if RIGHT_HIP_INDEX < len(kpts) else np.array([0, 0])



            #角度とスクリーン座標を計算する（正しいピッチ方向）
            pitch, yaw = calculate_head_pose(left_eye, right_eye, nose)
            screen_x, screen_y = calculate_screen_point(pitch, yaw, SCREEN_DISTANCE)

            #画像サイズにマッピング
            img_h, img_w = frame.shape[:2]
            traj_x = int((screen_x + 1) * img_w / 2)
            traj_y = int((1 - screen_y) * img_h / 2)
            trajectory.append((traj_x, traj_y))

            if len(trajectory) > TRAIL_LENGTH:
                trajectory.pop(0)


            # 芸術的なアプローチで軌跡を描く
            #frame_pose = draw_artistic_trail(frame_pose, trajectory)

            cv2.putText(frame_pose, f"Pitch: {pitch:.1f}°  Yaw: {yaw:.1f}°", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_pose, f"Screen: X={screen_x:.2f}m, Y={screen_y:.2f}m", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            safe_draw_line(frame_pose, left_eye, right_eye, RED, 2)
            safe_draw_line(frame_pose, right_eye, nose, BLUE, 2)
            safe_draw_line(frame_pose, nose, left_eye, BLUE, 2)
            safe_draw_line(frame_pose, right_eye, right_shoulder, LIGHT_BLUE, 2)
            safe_draw_line(frame_pose, left_eye, left_shoulder, LIGHT_BLUE, 2)
            safe_draw_line(frame_pose, right_shoulder, right_elbow, YELLOW, 2)
            safe_draw_line(frame_pose, right_elbow, right_wrist, YELLOW, 2)
            safe_draw_line(frame_pose, left_shoulder, left_elbow, YELLOW, 2)
            safe_draw_line(frame_pose, left_elbow, left_wrist, YELLOW, 2)
            safe_draw_line(frame_pose, right_shoulder, right_hip, PURPLE, 2)
            safe_draw_line(frame_pose, left_shoulder, left_hip, PURPLE, 2)
            safe_draw_line(frame_pose, right_shoulder, left_shoulder, PURPLE, 2)
            safe_draw_line(frame_pose, right_hip, left_hip, PURPLE, 2)




            # キーポイントのマッピング
            frame_pose = draw_joints_with_names(frame_pose, kpts, KP_NAMES,
                                                [NOSE_INDEX, LEFT_EYE_INDEX, RIGHT_EYE_INDEX, LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX
                                                ,LEFT_ELBOW_INDEX, RIGHT_ELBOW_INDEX, LEFT_WRIST_INDEX, RIGHT_WRIST_INDEX,LEFT_HIP_INDEX, RIGHT_HIP_INDEX])


        cv2.imshow("Head Pose Estimation", frame_pose)

        #フレーム処理時間の計算
        frame_time = time.time() - start_time
        if not start_measurement:
            frame_count += 1
            if frame_count >= WARMUP_FRAME:
                print("Warming up complete. Starting FPS measurement.")
                start_measurement = True
                frame_count = 0
                total_time = 0
        else:
            total_time += frame_time
            frame_count += 1

        key = cv2.waitKey(10)
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            trajectory = []  # トラックを空にする

    #平均FPS
    if frame_count > 0:
        average_fps = frame_count / total_time
        # 平均FPS出力（測定が行われていれば）
        if start_measurement and frame_count > 0 and total_time > 0:
            average_fps = frame_count / total_time
            print(f"Average FPS: {average_fps:.2f}")
        else:
            print("FPS計測が行われませんでした。")

    cap.stop_acquisition()
    cap.close_device()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
