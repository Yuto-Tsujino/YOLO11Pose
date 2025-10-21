# draw_3d_pose.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import config # config.py をインポート
import os # osモジュールをインポート (ファイルパス操作やディレクトリ作成のため)

# --- FFmpegのパス設定 (必要な場合) ---
# FFmpegがシステムパスに見つからない場合に、実行ファイルのフルパスを明示的に指定します。
# ご自身の環境に合わせてパスを修正してください。
# 通常、FFmpegをインストールし、システムパスを通せばこの設定は不要です。
# 例 (Windows):
# plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'
# 例 (Linux/macOS, where ffmpeg is typically in /usr/local/bin or /opt/homebrew/bin etc.):
# plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg' # or your actual path

# --- 設定 ---
CSV_FILE_PATH = "output_keypoints_3d.csv"
NUM_KEYPOINTS = len(config.KP_NAMES) # config.KP_NAMES がインデックスと名前のマッピング辞書であることを期待

# 出力ファイル名とディレクトリ
OUTPUT_ANIMATION_DIR = "animations" # 保存先ディレクトリ名
MP4_FILENAME = "3d_pose_animation.mp4"
# GIF_FILENAME = "3d_pose_animation.gif" # GIF保存も行う場合

# 骨格の接続情報
SKELETON_CONNECTIONS = []
if hasattr(config, 'body_bones') and hasattr(config, 'KP_NAMES'):
    name_to_idx = {name: idx for idx, name in config.KP_NAMES.items()}
    for bone_def in config.body_bones:
        if len(bone_def) == 3:
            kp_name1, kp_name2, _ = bone_def
            idx1 = name_to_idx.get(kp_name1)
            idx2 = name_to_idx.get(kp_name2)
            if idx1 is not None and idx2 is not None:
                SKELETON_CONNECTIONS.append((idx1, idx2))
            else:
                if idx1 is None: print(f"警告: 関節名 '{kp_name1}' が config.KP_NAMES に見つかりません。")
                if idx2 is None: print(f"警告: 関節名 '{kp_name2}' が config.KP_NAMES に見つかりません。")
        else:
            print(f"警告: config.body_bones の定義形式が不正です: {bone_def}")
else:
    if not hasattr(config, 'body_bones'): print("警告: config.py に body_bones が定義されていません。")
    if not hasattr(config, 'KP_NAMES'): print("警告: config.py に KP_NAMES が定義されていません。")


# --- データ読み込みと前処理 ---
try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"エラー: {CSV_FILE_PATH} が見つかりません。")
    exit()

for col in ['X_3d', 'Y_3d', 'Z_3d']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

all_frames_data_raw = []
unique_frame_ids = sorted(df['frame_id'].unique())

for frame_id in unique_frame_ids:
    frame_df = df[df['frame_id'] == frame_id]
    keypoints_for_frame = np.full((NUM_KEYPOINTS, 3), np.nan)
    for _, row in frame_df.iterrows():
        kp_idx = int(row['kp_index'])
        if 0 <= kp_idx < NUM_KEYPOINTS:
            keypoints_for_frame[kp_idx, 0] = row['X_3d']
            keypoints_for_frame[kp_idx, 1] = row['Y_3d']
            keypoints_for_frame[kp_idx, 2] = row['Z_3d']
    all_frames_data_raw.append(keypoints_for_frame)

if not all_frames_data_raw:
    print("エラー: CSVファイルから読み込めるデータがありませんでした。")
    exit()

data_raw = np.array(all_frames_data_raw)
data_mpl = np.zeros_like(data_raw)
data_mpl[:, :, 0] = data_raw[:, :, 0]
data_mpl[:, :, 1] = data_raw[:, :, 2]
data_mpl[:, :, 2] = data_raw[:, :, 1]

print(f"Matplotlib plot data shape: {data_mpl.shape}")
print(f"Skeleton connections: {SKELETON_CONNECTIONS}")


# --- Matplotlib アニメーション設定 ---
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')

# 軸範囲設定
valid_data_mpl_flat = data_mpl[~np.isnan(data_mpl)]
if valid_data_mpl_flat.size > 0:
    margin_ratio = 0.2
    min_range_abs_half = 200

    x_coords_valid = data_mpl[~np.isnan(data_mpl[:, :, 0]), 0]
    if x_coords_valid.size > 0:
        x_min_mpl, x_max_mpl = np.nanmin(data_mpl[:, :, 0]), np.nanmax(data_mpl[:, :, 0])
        x_range = x_max_mpl - x_min_mpl if x_max_mpl > x_min_mpl else 0
        x_margin = max(x_range * margin_ratio, min_range_abs_half)
        ax.set_xlim([x_min_mpl - x_margin, x_max_mpl + x_margin])
    else:
        ax.set_xlim([-1000, 1000])

    y_coords_valid = data_mpl[~np.isnan(data_mpl[:, :, 1]), 1]
    if y_coords_valid.size > 0:
        y_min_mpl, y_max_mpl = np.nanmin(data_mpl[:, :, 1]), np.nanmax(data_mpl[:, :, 1])
        y_range = y_max_mpl - y_min_mpl if y_max_mpl > y_min_mpl else 0
        y_margin = max(y_range * margin_ratio, min_range_abs_half)
        ax.set_ylim([y_min_mpl - y_margin, y_max_mpl + y_margin])
    else:
        ax.set_ylim([0, 2000])

    z_coords_valid = data_mpl[~np.isnan(data_mpl[:, :, 2]), 2]
    if z_coords_valid.size > 0:
        z_min_mpl, z_max_mpl = np.nanmin(data_mpl[:, :, 2]), np.nanmax(data_mpl[:, :, 2])
        z_range = z_max_mpl - z_min_mpl if z_max_mpl > z_min_mpl else 0
        z_margin = max(z_range * margin_ratio, min_range_abs_half)
        ax.set_zlim([z_min_mpl - z_margin, z_max_mpl + z_margin])
    else:
        ax.set_zlim([-1000, 1000])
else:
    ax.set_xlim([-1000, 1000]); ax.set_ylim([0, 2000]); ax.set_zlim([-1000, 1000])


ax.set_xlabel('X (カメラ右方向)', fontsize=12)
ax.set_ylabel('Z (カメラ前方)', fontsize=12)
ax.set_zlabel('Y (カメラ下方向)', fontsize=12)
ax.invert_zaxis()
ax.view_init(elev=20, azim=-70)

JOINT_COLOR = 'red'
LINE_COLOR = 'blue'

scat = ax.scatter([], [], [], s=40, c=JOINT_COLOR, marker='o')
lines = []
if SKELETON_CONNECTIONS:
    for _ in SKELETON_CONNECTIONS:
        line, = ax.plot3D([], [], [], color=LINE_COLOR, lw=2.5)
        lines.append(line)

# --- update関数 ---
def update(frame_num):
    ax.set_title(f"Frame {frame_num + 1}/{len(data_mpl)}", fontsize=14)
    current_coords_mpl = data_mpl[frame_num]
    valid_mask = ~np.isnan(current_coords_mpl).any(axis=1)
    plot_coords_mpl = current_coords_mpl[valid_mask]

    if plot_coords_mpl.shape[0] > 0:
        scat._offsets3d = (plot_coords_mpl[:, 0], plot_coords_mpl[:, 1], plot_coords_mpl[:, 2])
    else:
        scat._offsets3d = ([], [], [])

    if SKELETON_CONNECTIONS and lines:
        for i, (idx1, idx2) in enumerate(SKELETON_CONNECTIONS):
            if i < len(lines):
                if not (np.isnan(current_coords_mpl[idx1]).any() or np.isnan(current_coords_mpl[idx2]).any()):
                    p1 = current_coords_mpl[idx1]
                    p2 = current_coords_mpl[idx2]
                    lines[i].set_data_3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
                else:
                    lines[i].set_data_3d([], [], [])
            else:
                print(f"Warning: Mismatch SKELETON_CONNECTIONS and lines list size at index {i}")
    return [scat] + lines

# --- アニメーション作成 ---
ani = animation.FuncAnimation(fig, update, frames=len(data_mpl), interval=50, blit=True)

# --- アニメーションの保存 ---
# 保存先ディレクトリを作成
if not os.path.exists(OUTPUT_ANIMATION_DIR):
    try:
        os.makedirs(OUTPUT_ANIMATION_DIR)
        print(f"ディレクトリを作成しました: {OUTPUT_ANIMATION_DIR}")
    except OSError as e:
        print(f"ディレクトリの作成に失敗しました: {OUTPUT_ANIMATION_DIR} ({e})")
        # エラーが発生した場合、カレントディレクトリに保存するなどのフォールバックも検討可能
        OUTPUT_ANIMATION_DIR = "." # カレントディレクトリにフォールバック

output_mp4_path = os.path.join(OUTPUT_ANIMATION_DIR, MP4_FILENAME)
# output_gif_path = os.path.join(OUTPUT_ANIMATION_DIR, GIF_FILENAME) # GIFも保存する場合

try:
    print(f"アニメーションをMP4動画として保存中: {output_mp4_path} (時間がかかる場合があります)")
    ani.save(output_mp4_path, writer='ffmpeg', fps=20, dpi=150,
             progress_callback=lambda i, n: print(f'MP4フレーム保存中: {i+1}/{n}'))
    print(f"MP4動画の保存が完了しました: {output_mp4_path}")
except RuntimeError as e: # RuntimeErrorはFFmpegが見つからない場合などに出やすい
    print(f"MP4動画の保存中にRuntimeErrorが発生しました: {e}")
    print("FFmpegがインストールされ、システムパスが通っているか、")
    print("またはスクリプト冒頭の plt.rcParams['animation.ffmpeg_path'] が正しく設定されているか確認してください。")
    print("エラーメッセージに 'unknown file extension' が含まれる場合もFFmpeg関連の問題の可能性が高いです。")
except Exception as e:
    print(f"MP4動画の保存中に予期せぬエラーが発生しました: {e}")
# --- 別の視点でアニメーションを保存 ---
print("\n別の視点でアニメーションを保存します。")

# 新しい視点を設定します。elev (仰角) と azim (方位角) を調整してください。
new_elev = 20  # 元の20から変更
new_azim = 10 # 元の-70から変更

print(f"新しい視点を設定: elev={new_elev}, azim={new_azim}")
ax.view_init(elev=new_elev, azim=new_azim)

# 新しいファイル名 (視点情報を含めると分かりやすいです)
# ファイル名が長くなりすぎる場合は適宜調整してください
# MP4_FILENAME_VIEW2 = f"3d_pose_animation_elev{new_elev}_azim{new_azim}.mp4"
MP4_FILENAME_VIEW2 = "3d_pose_animation_view2.mp4" # シンプルなファイル名
output_mp4_path_view2 = os.path.join(OUTPUT_ANIMATION_DIR, MP4_FILENAME_VIEW2)

try:
    print(f"アニメーションをMP4動画として保存中 (別視点): {output_mp4_path_view2} (時間がかかる場合があります)")
    # 同じ ani オブジェクトを使用して保存します。
    # FuncAnimationはsaveメソッドが呼ばれるたびに、その時点のFigureとAxesの設定に基づいてフレームを生成します。
    ani.save(output_mp4_path_view2, writer='ffmpeg', fps=20, dpi=150,
             progress_callback=lambda i, n: print(f'MP4フレーム保存中 (別視点): {i+1}/{n}'))
    print(f"MP4動画の保存が完了しました (別視点): {output_mp4_path_view2}")
except RuntimeError as e:
    print(f"MP4動画の保存中 (別視点) にRuntimeErrorが発生しました: {e}")
    print("FFmpegがインストールされ、システムパスが通っているか、")
    print("またはスクリプト冒頭の plt.rcParams['animation.ffmpeg_path'] が正しく設定されているか確認してください。")
except Exception as e:
    print(f"MP4動画の保存中 (別視点) に予期せぬエラーが発生しました: {e}")

# --- 表示 ---
# ファイル保存が主目的なら、plt.show() はコメントアウトしても良い
plt.show()