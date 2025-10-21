import config
import camera_handler
import pose_estimate
import visualization
import utils
import cv2
import numpy as np
import torch
import time
import threading
import queue
import csv
import os
from collections import deque
from h36m_mapping import (
    coco_array_to_dict, coco17_to_h36m17_xyconf,
    draw_h36m_skeleton, csv_rows_h36m,
    H36M17_NAMES, default_h36m_metadata
)

# === 可視化 / 保存のスイッチ（ここだけで切替可能） ===
SHOW_COCO_SKELETON = False     # YOLO(COCO)骨格の描画（既定: OFF）
SHOW_H36M_SKELETON = True      # H36M-17骨格の描画（既定: ON）

SHOW_H36M_LABELS   = True

WRITE_COCO_CSV     = False     # COCO形式CSVの書き出し（既定: OFF）
WRITE_H36M_CSV     = True      # H36M形式CSVの書き出し（既定: ON）

# 表示色（好みで変更可）
COCO_COLOR = (0, 255, 255)     
H36M_COLOR = (0, 165, 255)
H36M_LABEL_COLOR = (255, 0, 0)     

# キューを定義
frame_queue = queue.Queue(maxsize = 1)


# =================メイン処理 ==================
def main():

    frame_id = 0

    # 2Dキーポイントをメモリに保存するリスト
    all_keypoints_data = []

    pseudo_3d_data = [] # 各フレームで[(x, y, z), (x, y, z),...]
    
    #FPS計算のための変数
    WARMUP_FRAME = 10
    frame_count = 0
    total_time = 0
    start_measurement = False

    model = pose_estimate.load_model(config.MODEL_PATH)

    cap, image = camera_handler.initialize_camera()

    trajectory = []
    time.sleep(1)  # 少し待機してから画像を取得
    
    threading.Thread(target = camera_handler.camera_thread_func, args = (cap, image, frame_queue), daemon = True).start()
    
    print("Starting main loop")
    while True:
        start_time = time.time()
        frame = None

        try:
            frame = frame_queue.get(timeout=1) # 待機して取得
        except queue.Empty:
            print("Frame queue timeout.")
            continue

        if config.camera_matrix is None or config.dist_coeffs is None:
            print("キャリブレーションデータが未定義のため、処理を終了します")
            break
    
        imgsz = 640
        results = model(frame, imgsz, verbose=False)
  
        frame_pose = frame.copy()
        img_h, img_w = frame_pose.shape[:2]

        # 関節インデックス
        current_frame_3d_data = [None] * len(config.KP_NAMES)

        for result in results:
            if result.keypoints is None:
                continue

            kpts = result.keypoints.xy[0].cpu().numpy()  # CPUに戻して処理
            
            # リストにデータを追加
            for kp_idx in range(len(config.KP_NAMES)):
                kp_name = config.KP_NAMES.get(kp_idx, f"kp_{kp_idx}")
                if kp_idx < len(kpts) and not np.all(kpts[kp_idx] == 0):
                    coords_2d = kpts[kp_idx]
                    all_keypoints_data.append((frame_id, kp_idx, coords_2d[0], coords_2d[1]))
                else:
                    # 検出されなかったキーポイントもNaNの代わりに0.0として追加
                    all_keypoints_data.append((frame_id, kp_idx, 0.0, 0.0))

            current_joints_2d_map_indexed = {}
            for i in range(len(config.KP_NAMES)):
                if i < len(kpts):
                    if np.all(kpts[i] == 0):
                        current_joints_2d_map_indexed[i] = None
                    else:
                        current_joints_2d_map_indexed[i] = kpts[i]
                else:
                    current_joints_2d_map_indexed[i] = None


            # 描画処理
            kpts_for_draw = {}
            for name, idx in config.__dict__.items():
                if name.endswith("_INDEX"):
                    kp_name_str = name.replace("_INDEX", "").lower()
                    if idx in current_joints_2d_map_indexed and current_joints_2d_map_indexed[idx] is not None:
                        coords_np = current_joints_2d_map_indexed[idx]
                        if isinstance(coords_np, np.ndarray):
                            kpts_for_draw[kp_name_str] = coords_np.astype(int)
                        else:
                            kpts_for_draw[kp_name_str] = np.array(coords_np, dtype=int)

            kpts_xy = np.full((17, 2), np.nan, dtype=np.float32)
            for idx, name in config.KP_NAMES.items():
                pt = kpts_for_draw.get(name)
                if pt is not None:
                    kpts_xy[int(idx)] = (float(pt[0]), float(pt[1]))
            kpts_cf = None

            # COCO配列→COCO辞書
            coco_dict = coco_array_to_dict(kpts_xy, conf=kpts_cf)

            # COCO辞書→H36M-17
            h36m_xyc = coco17_to_h36m17_xyconf(coco_dict)

            # 描画
            if SHOW_H36M_SKELETON:
                draw_h36m_skeleton(frame_pose, h36m_xyc, color=H36M_COLOR, thickness=2)
            
            if SHOW_H36M_LABELS:
                indices_to_label = range(17)  # ← 全部に表示。必要なら絞ってOK

                for j in indices_to_label:
                    x, y, conf = float(h36m_xyc[j, 0]), float(h36m_xyc[j, 1]), float(h36m_xyc[j, 2])
                    if not (np.isfinite(x) and np.isfinite(y) and conf > 0.0):
                        continue  # 欠損はスキップ

                    name = H36M17_NAMES[j]
                    # 小さな点＋文字（見やすいように少し右上にオフセット）
                    cv2.circle(frame_pose, (int(x), int(y)), 3, H36M_LABEL_COLOR, -1)
                    cv2.putText(frame_pose, name, (int(x)+6, int(y)-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, H36M_LABEL_COLOR, 1, cv2.LINE_AA)
                    

            if SHOW_COCO_SKELETON:
                # 左腕
                if "left_shoulder" in kpts_for_draw and "left_elow" in kpts_for_draw:
                    utils.safe_draw_line(frame_pose, kpts_for_draw["left_shoulder"], kpts_for_draw["left_elbow"], config.YELLOW, 2)
                if "left_elbow" in kpts_for_draw and "left_wrist" in kpts_for_draw:
                    utils.safe_draw_line(frame_pose, kpts_for_draw["left_elbow"], kpts_for_draw["left_wrist"], config.YELLOW, 2)
                # 右腕
                if "right_shoulder" in kpts_for_draw and "right_elbow" in kpts_for_draw:
                    utils.safe_draw_line(frame_pose, kpts_for_draw["right_shoulder"], kpts_for_draw["right_elbow"], config.YELLOW, 2)
                if "right_elbow" in kpts_for_draw and "right_wrist" in kpts_for_draw:
                    utils.safe_draw_line(frame_pose, kpts_for_draw["right_elbow"], kpts_for_draw["right_wrist"], config.YELLOW, 2)
                # 左胴
                if "left_shoulder" in kpts_for_draw and "left_hip" in kpts_for_draw:
                    utils.safe_draw_line(frame_pose, kpts_for_draw["left_shoulder"], kpts_for_draw["left_hip"], config.YELLOW, 2)
                # 右胴
                if "right_shoulder" in kpts_for_draw and "right_hip" in kpts_for_draw:
                    utils.safe_draw_line(frame_pose, kpts_for_draw["right_shoulder"], kpts_for_draw["right_hip"], config.YELLOW, 2)
                # 肩幅
                if "left_shoulder" in kpts_for_draw and "right_shoulder" in kpts_for_draw:
                    utils.safe_draw_line(frame_pose, kpts_for_draw["left_shoulder"], kpts_for_draw["right_shoulder"], config.YELLOW, 2)
                # 腰幅
                if "left_hip" in kpts_for_draw and "right_hip" in kpts_for_draw:
                    utils.safe_draw_line(frame_pose, kpts_for_draw["left_hip"], kpts_for_draw["right_hip"], config.YELLOW, 2)
                # 首
                if "nose" in kpts_for_draw and "left_shoulder" in kpts_for_draw and "right_shoulder" in kpts_for_draw:
                    utils.safe_draw_line(frame_pose, kpts_for_draw["nose"], (kpts_for_draw["left_shoulder"] + kpts_for_draw["right_shoulder"]) / 2, config.YELLOW, 2)

                # 主要な関節名の描画
                indices_to_draw_names = [
                    config.NOSE_INDEX, config.LEFT_EYE_INDEX, config.RIGHT_EYE_INDEX, 
                    config.LEFT_EAR_INDEX, config.RIGHT_EAR_INDEX,
                    config.LEFT_SHOULDER_INDEX, config.LEFT_ELBOW_INDEX, config.LEFT_WRIST_INDEX,
                    config.RIGHT_SHOULDER_INDEX, config.RIGHT_ELBOW_INDEX, config.RIGHT_WRIST_INDEX,
                    config.LEFT_HIP_INDEX, config.RIGHT_HIP_INDEX
                ]
                kpts_for_name_drawing = {idx: current_joints_2d_map_indexed.get(idx) for idx in indices_to_draw_names if current_joints_2d_map_indexed.get(idx) is not None}
                
                relevant_kpts_for_drawing_names = np.array([val for val in kpts_for_name_drawing.values() if val is not None])
                if relevant_kpts_for_drawing_names.size > 0 :
                    for idx, coords in kpts_for_name_drawing.items():
                        if coords is not None:
                            cv2.putText(frame_pose, config.KP_NAMES[idx], 
                                        (int(coords[0])+5, int(coords[1])), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.GREEN, 1)
                            cv2.circle(frame_pose, (int(coords[0]), int(coords[1])), 3, config.RED, -1)

            # 頭部姿勢推定と軌跡描画 (元のコードから流用)
            nose_2d = current_joints_2d_map_indexed.get(config.NOSE_INDEX)
            left_eye_2d = current_joints_2d_map_indexed.get(config.LEFT_EYE_INDEX)
            right_eye_2d = current_joints_2d_map_indexed.get(config.RIGHT_EYE_INDEX)

            if nose_2d is not None and left_eye_2d is not None and right_eye_2d is not None:
                pitch, yaw = pose_estimate.calculate_head_pose(left_eye_2d, right_eye_2d, nose_2d)
                screen_x, screen_y = pose_estimate.calculate_screen_point(pitch, yaw, config.SCREEN_DISTANCE)

                img_h, img_w = frame_pose.shape[:2]
                traj_x = int((screen_x + 1) * img_w / 2)
                traj_y = int((1 - screen_y) * img_h / 2) # Y軸反転
                trajectory.append((traj_x, traj_y))

                if len(trajectory) > config.TRAIL_LENGTH:
                    trajectory.pop(0)
                
                # 軌跡描画 (元の draw_artistic_trail のようなものがあれば)
                for i in range(1, len(trajectory)):
                    if trajectory[i-1] is None or trajectory[i] is None:
                        continue
                    cv2.line(frame_pose, trajectory[i-1], trajectory[i], config.CYAN, 2)


                cv2.putText(frame_pose, f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame_pose, f"Screen: X={screen_x:.2f}m, Y={screen_y:.2f}m", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


            # 最初の1人処理したらループ抜ける (複数人対応しない場合)
            break 

        cv2.imshow("Pose Estimation with 3D", frame_pose)

        frame_id += 1

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
    #ループ終了後の処理
    print("Main Loop Finish")

    # 画像サイズ
    print(f"screen size: Width = {img_w}, Height = {img_h}")

    #平均FPS
    if frame_count > 0:
        average_fps = frame_count / total_time
        # 平均FPS出力（測定が行われていれば）
        if start_measurement and frame_count > 0 and total_time > 0:
            average_fps = frame_count / total_time
            print(f"Average FPS: {average_fps:.2f}")
        else:
            print("FPS計測が行われませんでした。")
    
    if cap is not None: # capが初期化されていれば
        # camera_handler.stop_camera_thread() # もしそのような関数があれば
        # スレッドがデーモンなので、メインスレッド終了時に自動終了するが、
        # リソース解放は明示的に行うのが望ましい
        if hasattr(cap, 'stop_acquisition') and callable(getattr(cap, 'stop_acquisition')):
            cap.stop_acquisition()
        if hasattr(cap, 'close_device') and callable(getattr(cap, 'close_device')):
            cap.close_device()
        print("Camera resources released.")
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()

    