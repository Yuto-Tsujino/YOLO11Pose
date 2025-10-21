import cv2
import numpy as np
import os
from ximea import xiapi
import time

CHESSBOARD_SIZE = (10, 7)
SQUARE_SIZE = 24.0
SAVE_DIR = "calibration_images"
os.makedirs(SAVE_DIR, exist_ok=True)

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

objpoints = []
imgpoints = []

print("[INFO] XIMEAカメラを初期化中...")
cam = xiapi.Camera()
cam.open_device()
cam.set_exposure(int(30000))  # 露光時間を長めに
cam.set_param('gain', 0.0)
cam.set_param('auto_wb', 1)
cam.start_acquisition()
img = xiapi.Image()

frame_count = 0
MAX_FRAMES = 20
print("チェスボードをカメラに向けてください。's'キーで画像を保存、'q'で終了")

while frame_count < MAX_FRAMES:
    start_time = time.time()
    cam.get_image(img)
    frame = img.get_image_data_numpy()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[{frame_count}] frame取得時間: {elapsed:.4f} 秒")
    
    if end_time - start_time > 0.2:
        print("[WARN] フレーム取得が遅延しています")

    if frame.ndim == 2:
        gray = frame
        color_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        color_frame = frame.copy()

    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if found:
        cv2.drawChessboardCorners(color_frame, CHESSBOARD_SIZE, corners, found)
    else:
        cv2.putText(color_frame, "Chessboard NOT detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(color_frame, f"Saved frames: {frame_count}/{MAX_FRAMES}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("XIMEA Calibration", color_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and found:
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        filename = os.path.join(SAVE_DIR, f"calib_{frame_count:02d}.png")
        cv2.imwrite(filename, color_frame)
        print(f"[INFO] 画像{frame_count}枚目を保存: {filename}")
        frame_count += 1
        time.sleep(0.3)  # 手ブレ対策

    elif key == ord('q'):
        break

cam.stop_acquisition()
cam.close_device()
cv2.destroyAllWindows()

if len(objpoints) < 5:
    print("画像数が足りません（最低5枚推奨）")
    exit()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n=== キャリブレーション結果 ===")
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

np.savez("calibration_data.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print("\n[OK] calibration_data.npz に保存しました")
