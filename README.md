# YOLO-Pose Project

2D pose estimation pipeline using **Ultralytics YOLO**  
共通動作環境：**Windows (PowerShell + venv)** および **Linux (Conda + Python 3.10)**

---

## 🧩 1. Repository Structure
```
yolo-pose-project/
│
├── data/ # サンプル画像・動画（.gitignore 対象）
├── weights/ # 学習済みモデル（.gitignore 対象）
├── scripts/
│ ├── setup_linux_venv.sh # Linux 用自動セットアップスクリプト
│ └── check_env.ps1 # Windows 環境チェッカー
├── requirements_full.txt # 全依存パッケージ（PyTorch含む）
├── requirements_no_torch.txt # PyTorch 抜き（移植・軽量用）
├── .gitignore
└── yolo11n.pt # モデル重みファイル
```

---

## 🪟 2. Windows Setup (venv + PowerShell)

```powershell
# プロジェクトディレクトリに移動
cd C:\yolo-pose-project

# venv 有効化（事前に python -m venv yolo_env を作成済みと仮定）
.\yolo_env\Scripts\activate

# Python バージョン確認
python -V     # 例: Python 3.13 など

# 依存パッケージのインストール（PyTorch なし）
pip install -r requirements_no_torch.txt

# CUDA 版 PyTorch を追加（例: CUDA 12.6 に対応）
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio

# YOLO チェックと簡単な動作テスト
yolo checks
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo11n.pt")
print("✅ Ultralytics OK on Windows")
PY

# Conda 環境作成
conda create -n yolo_env python=3.10 -y
conda activate yolo_env

# 依存パッケージのインストール（PyTorch なし）
pip install -r requirements_no_torch.txt

# CUDA バージョン確認（cu121 など）
nvidia-smi   # → CUDA 12.1 なら cu121 を選択

# 対応する PyTorch をインストール
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# YOLO チェックと簡単なテスト
yolo checks
python - <<'PY'
from ultralytics import YOLO
YOLO("yolo11n.pt")
print("✅ Ultralytics OK on Linux (Conda 3.10)")
PY

# Pose 推論をテスト（Webカメラを使用）
yolo pose predict model=yolo11n.pt source=0 device=0


