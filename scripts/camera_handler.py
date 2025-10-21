from ximea import xiapi
import cv2
import numpy as np
import time
import config
 
def initialize_camera():
    cap = xiapi.Camera()
    try:
        cap.open_device()
        print("Camera successfully opened.")
        cap.start_acquisition()  # Acquisition（画像取得）を開始
        # 露出時間（マイクロ秒）→ 明るくするには大きく
        cap.set_exposure(config.XIMEA_EXPOSURE)  # 例：20000us = 20ms（デフォルトより明るめ）
        # ゲイン（増幅）→ ノイズも増えるけど暗さ対策になる
        cap.set_param('gain', config.XIMEA_GAIN)  # デフォルトは0.0、範囲は0〜最大値（カメラによる）
        # ホワイトバランスを自動にする（カラー画像が正しく見えるように）
        cap.set_param('auto_wb', config.XIMEA_AUTO_WB)
        print("Camera acquisition started.")
        return cap, xiapi.Image()
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def undistort_frame(frame):
    h, w = frame.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        config.camera_matrix, config.dist_coeffs, (w, h), 1, (w, h))
    return cv2.undistort(frame, config.camera_matrix, config.dist_coeffs, None, new_camera_matrix)

def camera_thread_func(cap, image, frame_queue):
    while True:
        try:
            cap.get_image(image)
            frame = image.get_image_data_numpy()
            if frame is not None and frame.size > 0:
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                frame = undistort_frame(frame)

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