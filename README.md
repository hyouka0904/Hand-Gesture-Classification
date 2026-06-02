# Hand Gesture Classification on Edge Devices
> CS3570 Multimedia Technology — Final Project
> Challenge from Microsoft | HaGRIDv2 · 2026

---

## Overview

一個 compact(≤ 10 MB)的 hand-gesture classifier，跑在 edge device 上。
模型吃 TA 提供的 preprocessing module 產生的 **cropped hand image** +
**21 個 MediaPipe landmark 座標**，輸出六個 class index 之一:

| Index | Label | 說明 |
|-------|-------|------|
| 0 | N/A | 其他所有手勢 / 未知 |
| 1 | fist | 握拳 |
| 2 | like | 讚(豎大拇指) |
| 3 | ok | OK 手勢 |
| 4 | one | 食指比一 |
| 5 | palm | 張開手掌 |

下游 classifier 是我們唯一要做的部分。hand detection 與 landmark extraction
是固定的，由 TA 提供(`hand_preprocess.py`，MediaPipe)。我們不會拿到
full-frame 影像。

---

## Repository Structure

```
.
├── inference.py              # Submission entry point (FIXED): predict(crop, landmarks) -> int
├── hand_preprocess.py        # TA-provided MediaPipe preprocessing. Local demo / data-prep only,
│                             #   NOT packed into the submission zip.
├── model/                    # Final weights for submission (spec-mandated zip folder, singular)
│   └── gesture_model.*       #   e.g. gesture_model.onnx or .pth, must be <= 10 MB
├── src/
│   ├── predictor.py          # Shared inference core: load weights + crop preprocessing + N/A heuristic
│   ├── dataset.py            # HaGRIDv2 + landmark-annotation loader
│   ├── train.py              # Shared training driver
│   ├── evaluate.py           # Shared evaluation / confusion matrix driver
│   ├── augmentation/         # Work split #1
│   ├── models/               # Work split #2 (plural — may hold multiple architectures)
│   └── compression/          # Work split #3
├── config/
│   ├── augmentation/         # YAML configs for augmentation pipelines
│   ├── models/               # YAML configs for model architectures / hyperparameters
│   └── compression/          # YAML configs for compression schemes
├── requirements.txt          # Inference-only deps (no mediapipe / no training deps)
├── requirements-train.txt    # Full training deps
├── README.md
└── .gitignore
```

> **`model/` vs `src/models/`** — 這是刻意的，不是筆誤。
> top-level **`model/`**(單數)是 spec 規定、放最終單一 weight 檔的資料夾。
> **`src/models/`**(複數)是 architecture 的程式碼，可以定義多個候選 model。

---

## The `predict()` Interface (read this carefully)

TA 的 evaluation harness 會先跑 preprocessing，再 `from inference import predict`
直接呼叫。**檔名 `inference.py` 與 `predict` 的 signature 是固定的** —— 不能改名、
也不能加參數，因為 harness 就是照原樣 import。

```python
import numpy as np

def predict(cropped_img: np.ndarray, landmarks: np.ndarray) -> int:
    """
    Args:
        cropped_img : np.ndarray, shape (H, W, 3), dtype uint8, RGB.
                      VARIABLE size — it is the hand bbox cropped with ~30%
                      padding by the TA preprocessor (see hand_preprocess.py).
        landmarks   : np.ndarray, shape (21, 2), dtype float32.
                      x, y ONLY (no z). Coordinates are normalized to [0, 1]
                      RELATIVE TO THE CROP BOX, not to the original image.
    Returns:
        int in {0, 1, 2, 3, 4, 5}
        0 = N/A, 1 = fist, 2 = like, 3 = ok, 4 = one, 5 = palm
    """
    ...
    return final_decision_class
```

**Exact I/O contract(對照 `hand_preprocess.py` 確認過):**

| Input | Shape | dtype | 說明 |
|-------|-------|-------|------|
| `cropped_img` | `(H, W, 3)` | `uint8` | RGB，**尺寸不固定**，bbox + 約 30% padding |
| `landmarks` | `(21, 2)` | `float32` | **只有 x, y、沒有 z**;正規化 [0,1]，**對 crop 相對** |

> preprocessor(`detect_hand`)只取 `point.x, point.y`，所以 classifier 永遠
> 收到 `(21, 2)` 的 array。**不要**設計任何依賴 z 座標的 feature 或 model branch。

`inference.py` 裡所有 path 都必須**相對於 `inference.py`**。

---

## Design Notes — How Each Side Should Be Built

### Image branch (the crop)
- crop 尺寸不固定，所以 `src/predictor.py` 必須先把它 resize 到固定輸入
  (例如 112×112)再進 backbone。
- 用 **pad-resize**(letterbox)保持長寬比，避免把手的形狀拉變形，之後做
  ImageNet normalization。
- Backbone:**MobileNetV3-Small**(ImageNet-pretrained，spec 允許;也能壓在
  10 MB 預算內)。

### Landmark branch (the 21 points)
- 輸入是 `(21, 2)` 的 crop 相對座標。為了 translation / scale invariance，
  建議以 **wrist(第 0 點)**為原點重新正規化，再除以手部 span。
- 兩種可用形式:攤平成 **42 維 vector → 小型 MLP**，或計算 **geometric
  feature**(手指關節角度 / 指尖距離)同時當作 N/A rule。
- **沒有 z 可用**，所以 depth-based feature 都不行。

### Fusion and N/A
- 把 image embedding 與 landmark embedding concat，後面接一個 6 類的
  classifier head。
- N/A **不是**單純取 network argmax。spec 鼓勵 engineering heuristic:用
  **softmax confidence threshold** 加上 **landmark-rule gate**(例如拒絕不合理
  的手部幾何)來壓 false trigger。
- 偏向 N/A:一次 false trigger 扣 **−2**，答對只 **+1**。

---

## Model / Compression Selection (config mechanism)

因為 `predict()` 不能加 selection 參數，各 work split 的變體選擇放在
`src/predictor.py` + `config/`:

```python
GesturePredictor(
    model_name="mobilenetv3_small",   # picks config/models/<name>.yaml
    compression_name="none",          # picks config/compression/<name>.yaml
    aug_name="default",               # provenance only (training-time)
)
```

- 每個部分都有 **default**，繳交的 `inference.py` 就用 default 實例化
  `GesturePredictor`。
- 每個 `config/<part>/*.yaml` 描述一個變體(augmentation pipeline 參數 /
  model architecture + hyperparameter / compression scheme)。
- augmentation 是 training-time 的事;它放進同一套 config schema 只是為了
  reproducibility，真正影響 inference 時所載入 weight 的只有 **model +
  compression**。

---

## Environment Setup

### 1. Create conda environment (Python 3.10)

```bash
conda create -n gesture-cls python=3.10 -y
conda activate gesture-cls
```

> **為什麼是 3.10?** Google Colab 的 default runtime 是 Python 3.10。
> MediaPipe 0.10.x 在 3.11 有 protobuf 衝突、在 3.12 wheel 不完整。3.10 對所有
> dependency 最穩，而且官方評分就是在 fresh Colab runtime 上跑。

### 2. Install dependencies

**Inference only**(對齊 Colab 繳交環境 —— 不含 MediaPipe，因為 preprocessing
由 TA 提供):
```bash
pip install -r requirements.txt
```

**Full training environment**:
```bash
pip install -r requirements-train.txt
```

### 3. GPU support (optional, training only)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Dataset

本專案使用 **HaGRIDv2 512px** 輕量版(`min_side = 512`，約 119.4 GB)，依 spec
規定。下載 image 與 landmark annotation:

```bash
# Images (512px lightweight version)
wget https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/hagridv2_512.zip
unzip hagridv2_512.zip -d data/hagridv2_512

# Annotations (contain MediaPipe hand_landmarks — usable directly as the 21-point input)
wget https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip
unzip annotations.zip -d data/annotations
```

建議的本機結構:
```
data/
├── hagridv2_512/
│   ├── fist/   like/   ok/   one/   palm/      # 5 target classes
│   └── <other one-handed classes>/             # N/A training pool (see below)
└── annotations/
    ├── train/   val/   test/                   # *.json per class, with hand_landmarks
```

### N/A class composition (important)

spec 規定 N/A 樣本**只用單手非目標手勢**，**雙手類別要排除**。HaGRIDv2 共有
33 個 gesture class:

- 5 個是 target(`fist`, `like`, `ok`, `one`, `palm`)。
- 7 個是**雙手** → **從 N/A 排除**:
  `hand_heart`, `hand_heart2`, `thumb_index2`, `timeout`, `holy`,
  `take_picture`, `xsign`。
- 剩下的 **21 個單手非目標** class **+ `no_gesture` class** 組成 N/A 的
  training pool。

> ⚠️ **不要**把 dataset commit 進 git。它已列在 `.gitignore`。

---

## Training

```bash
python -m src.train \
    --data_root data/hagridv2_512 \
    --ann_root  data/annotations \
    --model_cfg config/models/mobilenetv3_small.yaml \
    --aug_cfg   config/augmentation/default.yaml \
    --epochs 30 --batch_size 64 \
    --output_dir checkpoints/
```

---

## Evaluation Scoring (120 pts total)

| Criteria | Points | Rule |
|----------|--------|------|
| Model Size | 30 pts | `(10 − size_MB) × 3`;**超過 10 MB 直接 0 分** |
| Basic Performance (HaGRIDv2) | 20 pts | 按下表逐筆計分;`(RawScore / MaxRawScore) × 20` |
| Real-World Robustness | 40 pts | 按下表逐筆計分;`(RawScore / MaxRawScore) × 40` |
| Presentation | 30 pts | Live demo + 防禦機制說明 |

**逐情況計分(Basic Performance 與 Robustness 共用):**

| GT (ground truth) | Prediction | Score |
|-------------------|------------|-------|
| 5-class | correct class | **+1** |
| 5-class | wrong class / NA | **−2** |
| NA | any 5-class | **−2** |
| NA | NA | **0** |

**Rationale:** NA 若正確處理，視為「no reward / no penalty」(得 0 分);
只有在誤觸成一個 false event(把 NA 判成某個 5-class)時才扣分。

**其他影響設計的 spec 細節:**
- **false trigger 扣 2 倍** → 保守判 N/A 優於積極分類。
- **target 只在 standard pose 下計分。** ambiguous、distorted、非標準姿態都必須
  當成 **N/A**。
- Basic Performance 測試集有 **TA 加的 augmentation**(bbox jitter、blur 等) ——
  我們的 training augmentation 必須涵蓋這些。
- Real-World Robustness 是 **TA 自拍**的集合:**50 張 N/A 干擾影像(日常動作)
  + 50 張 target 影像**。

---

## Baseline References

以下是合規的參考方法(都不依賴 HaGRID-pretrained weight):

- **Image branch:** MobileNetV3 — A. Howard et al., *Searching for MobileNetV3*,
  ICCV 2019. arXiv:1905.02244。ImageNet-pretrained backbone，spec 明文允許。
- **Landmark branch / edge baseline:** F. Zhang et al., *MediaPipe Hands:
  On-device Real-time Hand Tracking*, 2020. arXiv:2006.10214。21 點 landmark 的
  來源;spec 允許。
- **Dataset:** A. Nuzhdin et al., *HaGRIDv2: 1M Images for Static and Dynamic
  Hand Gesture Recognition*, 2024. arXiv:2412.01508。

> 官方 HaGRIDv2 leaderboard 的 model 是 **HaGRID-pretrained，因此不可用** ——
> 這裡不使用，只列出作為對照。

---

## Submission

```bash
zip -r team_X.zip inference.py model/ src/ requirements.txt README.md
```

- `inference.py` 必須放在 zip 的**最上層**。
- 因為 `inference.py` 會 import `src/predictor.py`(以及所選的 `src/models/`
  module)，這些檔案**也必須一起打包**。請確認 zip 在 **fresh Colab runtime
  無需手動修改**即可解開並執行。
- `hand_preprocess.py` 由 TA 提供 —— **不要打包**。

---

## Notes

- 在 HaGRID / HaGRIDv2 上 pretrained 的 weight **不允許**。
  ImageNet-pretrained backbone(例如 MobileNetV3-Small)**允許**，
  **MediaPipe Hand Landmarker** 也允許。
- N/A 的 engineering heuristic(confidence thresholding、landmark-based rule)
  是 spec **明文鼓勵**的。
- Full-frame raw image **嚴格禁止**當作 model input。

---

## TODO

Work is split three ways: **augmentation · model · compression**.

### Spec alignment
- [ ] Build N/A pool from the 21 one-handed non-target classes + `no_gesture`; exclude the 7 two-handed classes
- [ ] Implement standard-pose-only handling for targets (ambiguous / distorted -> N/A)
- [ ] Confirm augmentation covers TA's bbox-jitter / blur on the basic set
- [ ] Sanity-check against the Robustness format (50 N/A daily actions + 50 targets)

### Augmentation (`src/augmentation/`, `config/augmentation/`)
- [ ] DataLoader: dynamic ±10% shift / scale / slight rotation on the crop
- [ ] Random Gaussian Blur + Color Jitter (low-light / motion noise)
- [ ] Optionally bbox jitter to mirror TA test-time augmentation

### Model (`src/models/`, `config/models/`)
- [ ] Backbone: MobileNetV3-Small / ShuffleNetV2 (multi-scale features)
- [ ] Attention: CBAM or SE module
- [ ] Loss: ArcFace / CosFace (widen angular margin between confusable classes)
- [ ] (optional) SimCLR / MoCo self-supervised pre-training on HaGRIDv2
- [ ] (optional) TSM / MoViNet light temporal modeling
- [ ] Landmark branch: wrist-relative normalization -> MLP / geometric features

### Compression (`src/compression/`, `config/compression/`)
- [ ] Pruning + retrain (threshold per layer, zeroed grads stay zero)
- [ ] K-means weight quantization + lookup table (shared centroids)
- [ ] Huffman coding on pruned CSR weights + indices
- [ ] Ref: Han et al., *Deep Compression*, ICLR 2016

### Infrastructure
- [ ] `src/predictor.py` shared core with `model_name` / `compression_name` selection + defaults
- [ ] Per-part multiple variants wired through `config/<part>/*.yaml`
- [ ] `inference.py` thin wrapper using default config; verify in fresh Colab
- [ ] Confirm final `model/` weights <= 10 MB

### Files still to review (currently empty)
- [ ] inference.py · src/predictor.py · src/dataset.py · src/train.py · src/evaluate.py · requirements.txt