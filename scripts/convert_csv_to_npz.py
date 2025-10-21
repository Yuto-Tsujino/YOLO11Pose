import pandas as pd
import numpy as np
import os

# 設定
csv_path = r'C:\yolo11_project1\output_keypoints.csv'
image_width = 1280
image_height = 480

# 出力先パスの構築
output_npz = r'C:\VideoPose3D\data\data_2d_h36m_custom_keypoints.npz'

# CSV読み込み
df = pd.read_csv(csv_path)
print(df.head())  # 最初の5行を表示
print(df.columns)  # 列名をすべて表示

required_cols = {'frame_id', 'kp_index', 'x_2d', 'y_2d'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSVに必要な列が含まれていません: {df.columns.tolist()}")

# NaNを0で埋め、confidenceも導出
df['x_2d'] = pd.to_numeric(df['x_2d'], errors='coerce').fillna(0)
df['y_2d'] = pd.to_numeric(df['y_2d'], errors='coerce').fillna(0)

# ピボットして(frames, keypoints, 2)の配列を作成
num_frames = df['frame_id'].max() + 1
num_keypoints = df['kp_index'].max() + 1

keypoints_array  =np.zeros((num_frames, num_keypoints, 3), dtype = np.float32)

for _, row in df.iterrows():
    f = int(row['frame_id'])
    k = int(row['kp_index'])
    x = float(row['x_2d'])
    y = float(row['y_2d'])
    conf = 1.0 if x != 0.0 and y != 0.0 else 0.0

    keypoints_array[f, k] = [x, y, conf]

# # # npz形式で保存
# data = {
#     'custom_video':{
#         'custom_subject': [{
#             'keypoints': keypoints_array,
#             'video_metadata': {
#                 'width': image_width,
#                 'height': image_height,
#                 'res_w': image_width,
#                 'res_h': image_height,
#             }
#         }]
#     }
# }

# VideoPose3Dが求める構造に変換
positions_2d = {
    'custom_subject': {
        'custom_video': [{
            'keypoints': keypoints_array,
            'video_metadata': {
                'width': image_width,
                'height': image_height,
                'res_w': image_width,
                'res_h': image_height,
                'fps' : 30
            }
        }]
    }
}

# metadata の追加（左右対称なキーポイントインデックスのペア）
# 例: COCOスタイル 17関節（順番に注意）
metadata = {
    'layout_name': 'coco',
    'num_joints': num_keypoints,
    'keypoints_symmetry': (
        [1, 3, 5, 7, 9, 11, 13, 15],  # left
        [2, 4, 6, 8, 10,12, 14, 16]   # right
    )
}

# 保存
os.makedirs(os.path.dirname(output_npz), exist_ok=True)
np.savez_compressed(output_npz, positions_2d=positions_2d, metadata=metadata)
print(f"[成功] VideoPose3D用 .npz 保存: {output_npz}")