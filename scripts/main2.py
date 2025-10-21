#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import csv
import threading
from collections import deque

import cv2
import numpy as np

# あなたのプロジェクト内モジュール
import config
import utils
import pose_estimate
import camera_handler

# >>> NEW: H3.6M 変換ユーティリティを読み込み
from h36m_mapping import (
    coco_array_to_dict, coco17_to_h36m17_xyconf,
    draw_h36m_skeleton, csv_rows_h36m,
)

# =============================================================================
# 設定
# =============================================================================
FRAME_QUEUE_MAXSIZE = 1
WARMUP_FRAMES = 10
SHOW_WINDOW_NAME = "Pose Estimation (H36M-17)"
CSV_PATH = os.path.abspath("output_keypoints.csv")
OUTPUT_NPZ_PATH = r"C:\VideoPose3D\data\data_2d_h36m_custom_keypoints.npz"

# =============================================================================
# CSV 初期化
# =============================================================================
csv_file = open(CSV_PATH, mode="w", newline="", encoding="utf-8")
csv_writer_2d = csv.writer(csv_file)
# >>> CHANGED: H36M-17 の順序・名称で保存する前提（ヘッダは従来通り）
csv_writer_2d.writerow(["frame_id", "kp_name", "kp_index", "x_2d", "y_2d"])

# >>> NEW: H36M-17 フレーム蓄積（NPZ用：各フレーム (17,3)）
h36m_frames = []

# =============================================================================
# 関数：NPZ 保存（H36M-17）
# =============================================================================
def save_h36m_npz(frames_xyc, img_w, img_h, fps, output_path):
    """
    frames_xyc: list of (17,3), shape (F,17,3)
    """
    positions = np.stack(frames_xyc, axis=0) if len(frames_xyc) > 0 else np.zeros((0,17,3), np.float32)
    positions_2d = {
        "custom_subject": {
            "custom_video": [{
                "keypoints": positions,
                "video_metadata": {
                    "width": int(img_w), "height": int(img_h),
                    "res_w": int(img_w), "res_h": int(img_h),
                    "fps": int(fps),
                }
            }]
        }
    }
    metadata = {
        "layout_name": "h36m",
        "num_joints": 17,
        "keypoints_symmetry": ([4,5,6,11,12,13], [1,2,3,14,15,16]),
    }
    np.savez_compressed(output_path, positions_2d=positions_2d, metadata=metadata)

# =============================================================================
# メイン
# =============================================================================
def main():
    # GPU/CPU 情報（任意）
    try:
        gpu_name = utils.get_gpu_name()
        print(f"Using GPU: {gpu_name}")
    except Exception:
        print("Using CPU or unknown GPU")

    # カメラ初期化
    cap = camera_handler.initialize_camera()
    if cap is None or not cap.isOpened():
        print("ERROR: Failed to open camera.")
        return

    print("Camera successfully opened.")

    # 表示ウィンドウ（リサイズ可能）
    cv2.namedWindow(SHOW_WINDOW_NAME, cv2.WINDOW_NORMAL)              # >>> NEW
    cv2.resizeWindow(SHOW_WINDOW_NAME, 1280, 720)                     # >>> NEW

    # カメラスレッド起動（最新フレームのみ保持）
    frame_queue = deque(maxlen=FRAME_QUEUE_MAXSIZE)
    image_lock = threading.Lock()
    image = {"frame": None}

    # >>> CHANGED: タイポ対策。camera_handler 側の関数名に合わせて取得
    cam_thread_target = getattr(camera_handler, "camera_thread_func",
                                getattr(camera_handler, "camere_thread_func", None))
    if cam_thread_target is None:
        raise AttributeError("camera_handler に camera_thread_func / camere_thread_func が見つかりません")

    print("Camera acquisition started.")
    threading.Thread(
        target=cam_thread_target,
        args=(cap, image, frame_queue, image_lock),
        daemon=True
    ).start()

    # モデル読み込み
    model = pose_estimate.load_model(config.MODEL_PATH)

    # カメラパラメータ（必須）
    if config.camera_matrix is None or config.dist_coeffs is None:
        print("Camera parameters are not set. Process will be terminated.")
        return

    print("Starting main loop")

    # FPS 計測の準備
    start_measurement = False
    warmup_count = 0
    frame_count = 0
    total_time = 0.0
    average_fps = 30  # フォールバック

    img_w = None
    img_h = None
    frame_id = 0
    trajectory = []  # 頭部向きの軌跡等に使っている場合

    # ウォームアップ
    while warmup_count < WARMUP_FRAMES:
        try:
            frame = frame_queue.pop()
        except IndexError:
            time.sleep(0.01)
            continue
        warmup_count += 1

    print("Warming up complete. Starting FPS measurement.")
    print("[INFO] Running... press 'q' on the preview window to quit.")
    start_measurement = True
    frame_count = 0
    total_time = 0.0

    # メイン処理ループ
    while True:
        # 最新フレーム取得
        try:
            frame = frame_queue.pop()
        except IndexError:
            # フレーム供給が途切れた場合の軽い待機
            time.sleep(0.01)
            continue

        if frame is None:
            continue

        if img_w is None or img_h is None:
            img_h, img_w = frame.shape[:2]

        t0 = time.perf_counter()

        # ---------------------------------------------------------------------
        # あなたの 2D 姿勢推定（YOLO / YOLOv8 Pose など）の処理：
        # - frame から 17点（COCO順）の2D座標を得る
        # - ここでは、既存の処理が "kpts_for_draw" に
        #   {"left_shoulder":(x,y), "right_shoulder":(x,y), ...} を入れている前提で流用
        #   （もし別名ならこの下の参照を合わせてください）
        # ---------------------------------------------------------------------
        # 例：
        results = pose_estimate.infer_keypoints(model, frame, imgsz=getattr(config, "IMG_SIZE", None))
        # 上の関数はダミー名です。あなたの実装名に合わせてください。
        # 以降で使うための dict を構成：
        kpts_for_draw = utils.extract_coco_kpt_dict(results, config)  # ← あなたの既存ヘルパに合わせてください
        # ---------------------------------------------------------------------

        # >>> NEW: COCO(=YOLO) 17 → H36M-17 変換
        # COCO (17,2) 配列を作成（未検出は NaN）
        yolo_xy = np.full((17, 2), np.nan, np.float32)
        for idx, name in config.KP_NAMES.items():
            pt = kpts_for_draw.get(name)
            if pt is not None and np.isfinite(pt[0]) and np.isfinite(pt[1]):
                yolo_xy[int(idx)] = (float(pt[0]), float(pt[1]))

        coco_dict = coco_array_to_dict(yolo_xy)
        h36m_xyc = coco17_to_h36m17_xyconf(coco_dict)

        # 表示用フレーム
        frame_pose = frame.copy()
        # >>> NEW: H36M-17 の骨格で描画（NaNは自動スキップ）
        draw_h36m_skeleton(frame_pose, h36m_xyc, color=config.YELLOW, thickness=2)

        # >>> CHANGED: ウィンドウ表示を有効化
        cv2.imshow(SHOW_WINDOW_NAME, frame_pose)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 終了
            break
        elif key == ord('c'):
            trajectory.clear()

        # >>> NEW: H36M-17 の順序・名称で CSV を保存
        for row in csv_rows_h36m(frame_id, h36m_xyc):
            csv_writer_2d.writerow(row)

        # >>> NEW: NPZ 用にフレームを貯める
        h36m_frames.append(h36m_xyc)

        # FPS 更新
        if start_measurement:
            dt = time.perf_counter() - t0
            total_time += dt
            frame_count += 1
            if frame_count % 60 == 0:
                inst_fps = 60.0 / max(total_time, 1e-6)
                print(f"[INFO] processed={frame_id} (+60 frames)  ~{inst_fps:.1f} FPS")
                average_fps = inst_fps  # 粗い推定でもOK
                total_time = 0.0

        frame_id += 1

    print("Main Loop Finish")

    if img_w is None or img_h is None:
        img_w = img_h = 0

    print(f"screen size: Width = {img_w}, Height = {img_h}")

    # >>> NEW: H36M-17 NPZ で保存（VideoPose3D互換）
    try:
        fps_to_write = int(average_fps) if np.isfinite(average_fps) else 30
    except Exception:
        fps_to_write = 30
    try:
        save_h36m_npz(h36m_frames, img_w, img_h, fps_to_write, OUTPUT_NPZ_PATH)
        print(f"Saved H36M-17 NPZ to: {OUTPUT_NPZ_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to save NPZ: {e}")

    # リソース解放
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
    csv_file.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
