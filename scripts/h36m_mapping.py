# h36m_mapping.py
# -*- coding: utf-8 -*-
"""
YOLO(COCO-17) 2D キーポイント → H3.6M-17 への変換ユーティリティ
- 入力： (17,2) の xy 配列（COCO順）と任意の信頼度 conf (17,) または None
- 出力： (17,3) の xy/conf（H36M-17 順）
- 欠損は NaN で扱い、描画やCSVでのNaN処理がしやすい形にします
"""

from typing import Iterable, Optional, Dict, Tuple, List
import numpy as np

# ===== H3.6M-17 の順序・名称（一般的な評価順）=====
H36M17_NAMES: List[str] = [
    "Pelvis", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "Spine1", "Neck", "Head", "Site",  # Site = 鼻（Nose）
    "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist",
]

# 左右対称（インデックス）
H36M_SYMMETRY = (
    [4, 5, 6, 11, 12, 13],  # left
    [1, 2, 3, 14, 15, 16],  # right
)

# 描画用エッジ（名称ペア）
H36M17_EDGES: List[Tuple[str, str]] = [
    # 脚
    ("Pelvis","RHip"),("RHip","RKnee"),("RKnee","RAnkle"),
    ("Pelvis","LHip"),("LHip","LKnee"),("LKnee","LAnkle"),
    # 体幹～頭
    ("Pelvis","Spine1"),("Spine1","Neck"),("Neck","Head"),("Head","Site"),
    # 腕
    ("Neck","LShoulder"),("LShoulder","LElbow"),("LElbow","LWrist"),
    ("Neck","RShoulder"),("RShoulder","RElbow"),("RElbow","RWrist"),
]

# ===== COCO-17（YOLO）名称 → インデックス =====
# あなたの config.KP_NAMES と一致させます（デフォルト想定）
COCO_NAME_TO_INDEX = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}

def _to_nan_xy(xy: Optional[Iterable[float]]) -> Tuple[float, float]:
    if xy is None:
        return (np.nan, np.nan)
    x, y = float(xy[0]), float(xy[1])
    return (x, y)

def _midpoint(a: Optional[Iterable[float]], b: Optional[Iterable[float]]) -> Optional[Tuple[float, float]]:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

def _extend_head(neck: Optional[Iterable[float]], nose: Optional[Iterable[float]], scale: float = 0.5
                 ) -> Optional[Tuple[float, float]]:
    """Head の近似：Neck→Nose ベクトルを延長（scale=0.3〜0.7で調整可）"""
    if neck is None or nose is None:
        return neck if neck is not None else None
    vx, vy = (nose[0] - neck[0]), (nose[1] - neck[1])
    return (nose[0] + vx * scale, nose[1] + vy * scale)

def coco_array_to_dict(
    kpts_xy: np.ndarray,
    conf: Optional[np.ndarray] = None,
    name_to_index: Dict[str, int] = COCO_NAME_TO_INDEX,
) -> Dict[str, Tuple[float, float]]:
    """
    (17,2) の COCO順 xy 配列を COCO 名称→(x,y) の dict にする（欠損は None）
    conf が与えられた場合、conf<=0 を欠損扱いにします
    """
    assert kpts_xy.shape[0] == 17 and kpts_xy.shape[1] == 2, "kpts_xy must be (17,2)"
    if conf is not None:
        conf = conf.reshape(-1)
        assert conf.shape[0] == 17, "conf must have length 17"

    out = {}
    for name, idx in name_to_index.items():
        xy = kpts_xy[idx]
        if conf is not None and (not np.isfinite(conf[idx]) or conf[idx] <= 0.0):
            out[name] = None
        elif not np.isfinite(xy[0]) or not np.isfinite(xy[1]):
            out[name] = None
        else:
            out[name] = (float(xy[0]), float(xy[1]))
    return out

def coco17_to_h36m17_xyconf(
    coco: Dict[str, Optional[Tuple[float, float]]],
    conf_by_name: Optional[Dict[str, float]] = None,
    head_extend_scale: float = 0.5,
) -> np.ndarray:
    """
    COCO dict → H36M-17 (17,3) の xy/conf 配列
    conf は xy が有効なら 1.0（または conf_by_name を使用）、欠損は 0.0
    """
    # 必要点の取得
    L_hip, R_hip = coco.get("left_hip"), coco.get("right_hip")
    L_sho, R_sho = coco.get("left_shoulder"), coco.get("right_shoulder")
    L_knee, R_knee = coco.get("left_knee"), coco.get("right_knee")
    L_ank, R_ank = coco.get("left_ankle"), coco.get("right_ankle")
    L_elb, R_elb = coco.get("left_elbow"), coco.get("right_elbow")
    L_wri, R_wri = coco.get("left_wrist"), coco.get("right_wrist")
    nose = coco.get("nose")

    pelvis = _midpoint(L_hip, R_hip)
    neck   = _midpoint(L_sho, R_sho)
    spine1 = _midpoint(pelvis, neck) if (pelvis is not None and neck is not None) else None
    head   = _extend_head(neck, nose, scale=head_extend_scale)
    site   = nose  # Site ≒ Nose

    mapping_xy = [
        pelvis,                 # 0 Pelvis
        R_hip, R_knee, R_ank,   # 1..3
        L_hip, L_knee, L_ank,   # 4..6
        spine1, neck, head,     # 7..9
        site,                   # 10 Site (Nose)
        L_sho, L_elb, L_wri,    # 11..13
        R_sho, R_elb, R_wri,    # 14..16
    ]

    # conf の決定
    xyconf = np.zeros((17, 3), dtype=np.float32)
    for j, xy in enumerate(mapping_xy):
        if xy is None:
            xyconf[j] = [np.nan, np.nan, 0.0]
        else:
            c = 1.0
            if conf_by_name is not None:
                # Pelvis/Neck/Spine1/Head は構成点の最小 conf を使うとよい
                if j == 0:   # Pelvis = mid(L/R hip)
                    c = min(conf_by_name.get("left_hip", 1.0), conf_by_name.get("right_hip", 1.0))
                elif j == 8: # Neck = mid(L/R shoulder)
                    c = min(conf_by_name.get("left_shoulder", 1.0), conf_by_name.get("right_shoulder", 1.0))
                elif j == 7: # Spine1 = mid(Pelvis,Neck)
                    c = c  # 既に 1.0（必要なら Pelvis/Neck の最小にしても良い）
                elif j == 9: # Head ≈ extend(Neck→Nose)
                    c = min(conf_by_name.get("nose", 1.0), conf_by_name.get("left_shoulder", 1.0),
                            conf_by_name.get("right_shoulder", 1.0))
                elif j == 10: # Site = nose
                    c = conf_by_name.get("nose", 1.0)
                # それ以外は対応する COCO 点の conf を使う
            xyconf[j] = [xy[0], xy[1], float(c)]
    return xyconf

def draw_h36m_skeleton(
    img, h36m_xyconf: np.ndarray, color=(0,255,255), thickness: int = 2
):
    """H36M-17 の骨格を線で可視化（safe: NaN を自動スキップ）"""
    import cv2
    name_to_idx = {n: i for i, n in enumerate(H36M17_NAMES)}
    for a, b in H36M17_EDGES:
        ia, ib = name_to_idx[a], name_to_idx[b]
        xa, ya = h36m_xyconf[ia, 0], h36m_xyconf[ia, 1]
        xb, yb = h36m_xyconf[ib, 0], h36m_xyconf[ib, 1]
        if np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb):
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), color, thickness)

def csv_rows_h36m(frame_id: int, h36m_xyconf: np.ndarray):
    """
    CSV 1フレーム分の行を生成： (frame_id, name, index, x, y)
    NaN は文字列 'NaN' で返します
    """
    rows = []
    for j, name in enumerate(H36M17_NAMES):
        x, y = h36m_xyconf[j, 0], h36m_xyconf[j, 1]
        if not np.isfinite(x) or not np.isfinite(y):
            rows.append([frame_id, name, j, "NaN", "NaN"])
        else:
            rows.append([frame_id, name, j, f"{x:.3f}", f"{y:.3f}"])
    return rows

def default_h36m_metadata(width: int, height: int, fps: int) -> Dict:
    """VideoPose3D 互換の metadata を返す"""
    return {
        "layout_name": "h36m",
        "num_joints": 17,
        "keypoints_symmetry": H36M_SYMMETRY,
        "video_metadata": {"width": width, "height": height, "res_w": width, "res_h": height, "fps": int(fps)},
    }


