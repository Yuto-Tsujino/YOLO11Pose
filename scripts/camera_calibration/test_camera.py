from ximea import xiapi
import time

cam = xiapi.Camera()

print("[INFO] カメラ初期化中...")
cam.open_device()
cam.set_imgdataformat("XI_MONO8")
cam.set_exposure(10000)  # 適度な露光
cam.set_width(640)
cam.set_height(480)
cam.start_acquisition()

img = xiapi.Image()
print("[INFO] 撮影開始")


for i in range(5):
    start = time.time()
    cam.get_image(img)
    frame = img.get_image_data_numpy()
    print(f"[{i}] frame shape: {frame.shape}, time: {time.time() - start:.3f}秒")

cam.stop_acquisition()
cam.close_device()
print("[INFO] テスト完了")
input("終了するにはEnterキーを押してください...")

