# Hand Gesture Classification on Edge Devices

> CS3570 Multimedia Technology — Final Project  
> Challenge from Microsoft | HaGRIDv2 · 2026

---

## Overview

本專案目標是做一個可部署在 edge device 上的 compact hand-gesture classifier，模型大小需控制在 10 MB 以內。

TA 提供 preprocessing module，負責：

```txt
full image
→ hand detection
→ cropped hand image
→ 21 hand landmark coordinates
```

我們只負責 downstream classifier。模型只會收到：

```txt
cropped_img : RGB hand crop
landmarks   : 21 個 2D hand landmarks
```

輸出六個 class 之一：

| Index | Label | 說明 |
|---:|---|---|
| 0 | N/A | 非目標手勢 / 未知 / 無效姿勢 |
| 1 | fist | 握拳 |
| 2 | like | 讚 / 大拇指 |
| 3 | ok | OK 手勢 |
| 4 | one | 食指比一 |
| 5 | palm | 張開手掌 |

hand detection 和 landmark extraction 不是本專案要做的部分。不能使用 full-frame raw image 作為 model input。

---

## Repository Structure

```txt
.
├── inference.py
├── hand_preprocess.py
├── train.py
├── build_mini_train.py
├── model/
│   └── gesture_model.ptmodel
├── checkpoints/
│   ├── gesture_model.pth
│   ├── gesture_model_pruned50.pth
│   └── gesture_model_pruned50_quant.pth
├── src/
│   ├── predictor.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── augmentation/
│   ├── models/
│   │   └── test.py
│   └── compression/
│       ├── compress.py
│       └── baseline.py
├── config/
│   ├── augmentation/
│   ├── models/
│   └── compression/
├── data/
│   └── test/
├── requirements.txt
├── requirements-train.txt
├── README.md
└── .gitignore
```

```txt
model/
```

放最終 submission 會使用的 model artifact，目前預設是：

```txt
model/gesture_model.ptmodel
```

```txt
src/models/
```

放 model architecture 定義。現在的 `src.models.test` 只是測 compression pipeline 的 placeholder model，不是最終 baseline model。

```txt
checkpoints/
```

放 training / compression 中間 checkpoint，不會放進最終 submission zip。

---

## Submission Interface

TA evaluator 會先跑 TA 提供的 preprocessing，再 import 我們的 `inference.py`：

```python
from inference import predict
```

`inference.py` 必須放在 zip 最上層，且必須實作：

```python
def predict(cropped_img: np.ndarray, landmarks: np.ndarray) -> int:
    return final_decision_class
```

Input contract：

| Input | Shape | dtype | 說明 |
|---|---|---|---|
| `cropped_img` | `(H, W, 3)` | `uint8` | RGB hand crop，尺寸不固定 |
| `landmarks` | `(21, 2)` | `float32` | x, y 座標，已 normalize 到 crop 座標系 |
| return | scalar | `int` | class id in `{0,1,2,3,4,5}` |

所有 `inference.py` 內的 path 都要相對於 `inference.py`。

目前 `inference.py` 預設載入：

```txt
model/gesture_model.ptmodel
```

---

## Model I/O Contract

training、compression、evaluation、inference 必須使用同一個 model forward signature：

```python
logits = model(crop, landmarks)
```

Tensor shape：

```txt
crop      : (B, 3, 112, 112)
landmarks : (B, 21, 2)
logits    : (B, 6)
```

`crop` 由 shared preprocessing 產生：

```txt
cropped_img
→ letterbox resize to 112 × 112
→ ImageNet normalization
→ CHW tensor
```

`landmarks` 維持 `(21, 2)`，若需要 wrist-relative normalization，應在 model 的 landmark branch 內處理。

---

## Model Design

預期架構是 dual-branch classifier：

```txt
image crop
→ image branch
→ image embedding
             \
              concat → classifier head → logits
             /
landmarks
→ landmark branch
→ landmark embedding
```

建議 final image branch：

```txt
MobileNetV3-Small
```

建議 landmark branch：

```txt
wrist-relative normalization
→ scale normalization
→ flatten 21 × 2 into 42-d vector
→ small MLP
```

目前 `src.models.test` 是 tiny dual-branch model，只用來測試：

```txt
.pth checkpoint 格式
model(crop, landmarks) I/O
compression pipeline
.ptmodel decode + inference
```

---

## N/A Handling

N/A 不是單純 network argmax。

目前 inference path 在：

```txt
src/predictor.py
```

決策流程：

```txt
logits
→ softmax
→ argmax
→ confidence threshold
→ landmark gate
→ final class
```

`conf_threshold` 解析順序：

```txt
explicit conf_threshold
→ .ptmodel meta["best_conf_threshold"]
→ fallback 0.5
```

`baseline.py` 會在 val set 上 sweep threshold，找出 spec raw score 最高的 threshold，存進 `.ptmodel` metadata。

目前 `landmark_gate()` 還是 no-op，永遠回傳 True。後續可加入：

```txt
finger joint angle sanity check
fingertip distance sanity check
hand span sanity check
invalid / distorted pose rejection
```

因為 false trigger 會扣 -2，所以 inference 應偏保守。

---

## Dataset

官方 dataset 是 HaGRIDv2 512px。

建議資料結構：

```txt
data/
├── hagridv2_512/
│   ├── fist/
│   ├── like/
│   ├── ok/
│   ├── one/
│   ├── palm/
│   └── ...
├── annotations/
│   ├── train/
│   ├── val/
│   └── test/
├── processed/          ← 全量 MediaPipe cache + packed .npy（自動產生）
└── mini_train/         ← mini subset（--mini_train 時自動產生，勿手動修改）
    ├── annotations/
    │   ├── train/
    │   └── val/
    └── processed/
```

若解壓後多包一層資料夾，要把內容移上來。

例如：

```txt
data/annotations/annotations/train
```

應改成：

```txt
data/annotations/train
```

例如：

```txt
data/hagridv2_512/HaGRIDv2_dataset_512/fist
```

應改成：

```txt
data/hagridv2_512/fist
```

Target classes：

```txt
fist
like
ok
one
palm
```

N/A 應使用 one-handed non-target gestures 加上 `no_gesture`。

two-handed classes 不應放入 N/A：

```txt
hand_heart
hand_heart2
thumb_index2
timeout
holy
take_picture
xsign
```

---

## Training

training entry：

```txt
train.py
```

輸入：`--data_root`（default `data`），底下的三個路徑自動推導：

```txt
image_root = <data_root>/hagridv2_512
ann_root   = <data_root>/annotations
cache_root = <data_root>/processed
```

輸出：

```txt
checkpoints/gesture_model.pth
```

checkpoint format：

```python
{
    "model_state_dict": ...,
    "model_cfg": ...,
    "label_map": ...,
    "val_acc": ...,
    "aug_cfg": ...
}
```

正式訓練（全量資料）：

```powershell
python train.py --epochs 30 --batch_size 64 --num_workers 4
```

Mini subset 快速迭代（每 class 2000 張，8:2 切，所有 cache 放在 `data/mini_train/`）：

```powershell
python train.py --mini_train --epochs 10 --batch_size 64 --num_workers 4
```

自訂 per-class 數量：

```powershell
python train.py --mini_train --mini_per_class 500 --epochs 5
```

自訂 data root：

```powershell
python train.py --data_root D:/datasets/hagrid --epochs 30
```

開啟 augmentation：

```powershell
python train.py --aug_cfg config/augmentation/default.yaml --epochs 30
```

`batch_size` 是 mini-batch size。  
例如 `batch_size=64` 代表每次 forward / backward 使用 64 筆樣本更新一次參數。

也可以獨立建立 mini subset 而不訓練：

```powershell
python build_mini_train.py
python build_mini_train.py --data_root D:/datasets/hagrid --per_class 1000
```

---

## Compression Overview

Compression 有兩條路線：

```txt
src/compression/compress.py
```

已測通的 pruning-only fallback：

```txt
.pth checkpoint
→ global magnitude unstructured pruning
→ fine-tune retraining
→ compressed .pth
→ compressed .onnx
```

```txt
src/compression/baseline.py
```

完整 Han et al. Deep Compression baseline：

```txt
.pth checkpoint
→ global magnitude unstructured pruning
→ fine-tune retraining
→ k-means weight sharing
→ centroid fine-tuning
→ Huffman coding
→ .ptmodel archive
```

主要 reference：

```txt
Han et al.,
Deep Compression: Compressing Deep Neural Networks with Pruning,
Trained Quantization and Huffman Coding,
ICLR 2016
```

---

## baseline.py Pipeline

`baseline.py` 是完整 compression baseline。

完整流程：

```txt
input .pth checkpoint

→ load checkpoint
→ build_model(model_cfg)
→ load model_state_dict

→ global magnitude pruning
→ prune retrain
→ save pruned .pth

→ per-layer 1-D k-means weight sharing
   Conv2d: 8-bit, 256 centroids
   Linear: 5-bit, 32 centroids

→ centroid fine-tuning
   assignment index fixed
   grouped gradient update only modifies centroid values
   model.eval() to freeze BN / dropout behavior

→ calibrate conf_threshold on val set
   sweep threshold
   maximize spec raw score

→ Huffman coding
   encode centroid index stream
   encode sparse structure using bitmask or relindex
   automatically choose smaller representation per layer

→ save .ptmodel
→ verify decode round-trip
```

Default input：

```txt
checkpoints/gesture_model.pth
```

Default outputs：

```txt
checkpoints/gesture_model_pruned50.pth
checkpoints/gesture_model_pruned50_quant.pth
model/gesture_model.ptmodel
```

---

## .ptmodel Format

`.ptmodel` 是 pickle-serialized Deep Compression archive，format key：

```txt
ptmodel-dc-v1
```

內容包含：

```python
{
    "format": "ptmodel-dc-v1",
    "model_cfg": {...},
    "label_map": ["N/A", "fist", "like", "ok", "one", "palm"],
    "compression": {
        "method": "deep_compression",
        "prune_amount": 0.5,
        "conv_bits": 8,
        "fc_bits": 5,
        "global_sparsity": 0.5,
        "best_conf_threshold": ...,
        ...
    },
    "tensors": {
        "<layer>.weight": {
            "kind": "q",
            "shape": [...],
            "bits": 8 or 5,
            "k": int,
            "n_weights": int,
            "centroids": np.float32 array,
            "code_lengths": np.uint8 array,
            "bitstream": bytes,
            "nbits": int,
            "sparse_enc": "bitmask" or "relindex",
            ...
        },
        "<other tensor>": {
            "kind": "raw",
            "array": np.ndarray
        }
    }
}
```

`.ptmodel` decoder 只依賴：

```txt
numpy
torch
Python stdlib
```

不需要：

```txt
scipy
sklearn
albumentations
mediapipe
onnxruntime
```

---

## Compression Commands

小資料快速測試：

```powershell
python -m src.compression.baseline --pth_in checkpoints/gesture_model.pth --ann_root data/test/annotations --image_root data/test/hagridv2_512 --cache_root data/test/processed --prune_epochs 1 --ft_epochs 1
```

完整預設測試：

```powershell
python -m src.compression.baseline --pth_in checkpoints/gesture_model.pth --ann_root data/test/annotations --image_root data/test/hagridv2_512 --cache_root data/test/processed
```

正式資料：

```powershell
python -m src.compression.baseline --pth_in checkpoints/gesture_model.pth --ann_root data/annotations --image_root data/hagridv2_512 --cache_root data/processed
```

成功 log 應包含：

```txt
[baseline] post-prune global sparsity=0.5000
[baseline] quantized Conv2d ...
[baseline] quantized Linear ...
[baseline] calibrated conf_threshold=...
[baseline] saved .ptmodel -> model\gesture_model.ptmodel
[baseline] decode round-trip OK
```

---

## Evaluation

evaluation entry：

```txt
src/evaluate.py
```

支援：

```txt
.pth
.onnx
.ptmodel
```

三種都透過 `GesturePredictor` 推論。

Evaluate original checkpoint：

```powershell
python -m src.evaluate --weights checkpoints/gesture_model.pth --cache_root data/test/processed --split val --conf_threshold 0
```

Evaluate compressed `.ptmodel`：

```powershell
python -m src.evaluate --weights model/gesture_model.ptmodel --cache_root data/test/processed --split val --conf_threshold 0
```

使用 `.ptmodel` 內校準 threshold：

```powershell
python -m src.evaluate --weights model/gesture_model.ptmodel --cache_root data/test/processed --split val
```

輸出 metrics：

```txt
model_size_mb
conf_threshold
confusion matrix
plain_accuracy
target_accuracy
na_false_trigger_rate
RawScore
MaxRawScore
score_ratio
Model Size score
Basic Performance estimate
Robustness estimate
```

---

## Local Compression Pipeline Test

這段只測 compression pipeline 是否能吃到正確 I/O，不代表模型真的好。

### 1. 準備小 HaGRID subset

資料夾結構：

```txt
data/test/
├── make_subset.py
├── annotations/
│   ├── train/
│   └── val/
├── hagridv2_512/
│   ├── fist/
│   ├── like/
│   ├── ok/
│   ├── one/
│   ├── palm/
│   └── no_gesture/
└── processed/
    ├── train/
    └── val/
```

建立 subset：

```powershell
python data/test/make_subset.py
```

建議 subset：

```txt
train: 每類 50 張
val: 每類 20 張
classes: fist, like, ok, one, palm, no_gesture
```

### 2. 訓練 tiny real-data test model

```powershell
python -m src.train --ann_root data/test/annotations --image_root data/test/hagridv2_512 --cache_root data/test/processed --epochs 1 --batch_size 16 --num_workers 0 --aug_cfg none --output_dir checkpoints
```

預期輸出：

```txt
checkpoints/gesture_model.pth
```

### 3. 跑完整 Deep Compression baseline

```powershell
python -m src.compression.baseline --pth_in checkpoints/gesture_model.pth --ann_root data/test/annotations --image_root data/test/hagridv2_512 --cache_root data/test/processed --prune_epochs 1 --ft_epochs 1
```

預期輸出：

```txt
checkpoints/gesture_model_pruned50.pth
checkpoints/gesture_model_pruned50_quant.pth
model/gesture_model.ptmodel
```

### 4. 評估原始 `.pth`

```powershell
python -m src.evaluate --weights checkpoints/gesture_model.pth --cache_root data/test/processed --split val --conf_threshold 0
```

### 5. 評估壓縮後 `.ptmodel`

```powershell
python -m src.evaluate --weights model/gesture_model.ptmodel --cache_root data/test/processed --split val --conf_threshold 0
```

若 pipeline 正確，兩邊 confusion matrix 應該接近。  
若 `.ptmodel` 的 decode round-trip 完全正確，壓縮後模型行為應與 quantized in-memory model 一致。

### 6. 測 submission import path

```powershell
python -c "from inference import predict; import numpy as np; print(predict(np.zeros((100,100,3),dtype='uint8'), np.zeros((21,2),dtype='float32')))"
```

預期輸出是一個 `{0,1,2,3,4,5}` 的 int。通常 dummy input 應該會是 `0`。

---

## ONNX Usage

spec 沒有要求 ONNX。

目前 final path 預設使用：

```txt
model/gesture_model.ptmodel
```

ONNX 只保留在 `compress.py` pruning-only fallback 裡，作為 optional deployment / debug output。  
完整 Deep Compression baseline 不以 ONNX 作為最終壓縮 artifact，因為 Huffman-coded / sparse-indexed representation 不是 standard ONNXRuntime 可以直接執行的 dense tensor graph。

---

## Design Trade-offs

### 為什麼使用 `.ptmodel`

`.ptmodel` 是本專案自訂 Deep Compression archive。  
它可以儲存：

```txt
sparse structure
centroid table
Huffman-coded centroid index stream
raw non-compressed tensors
compression metadata
```

這比普通 `.pth` 或 `.onnx` 更接近 Han et al. Deep Compression 的 storage representation。

### 為什麼 inference 可以吃 `.ptmodel`

`inference.py` 是我們自己提交的檔案。  
只要它能在 fresh Colab runtime 中載入 `model/gesture_model.ptmodel` 並正確回傳 `predict()` 結果，spec 沒有限制 model file 副檔名。

`.ptmodel` 第一次載入時會 decode 成 dense PyTorch state_dict，之後 cache model，不會每次 predict 都重新 decode。

### 為什麼不用 ONNX 當 full compression output

完整 Deep Compression 的 Huffman-coded artifact 本質上是 encoded storage，不是 standard ONNX graph。

若硬要輸出 ONNX，通常要先 decode / dequantize 回 dense tensor。  
這樣 ONNX 可以跑，但不代表 ONNX 本身保留了 Huffman-coded storage 優勢。

---

## Environment Setup

建議 Python：

```txt
Python 3.10
```

建立 conda environment：

```powershell
conda create -n gesture-cls python=3.10 -y
conda activate gesture-cls
```

安裝 inference dependencies：

```powershell
pip install -r requirements.txt
```

建議 `requirements.txt`：

```txt
numpy>=1.24
torch>=2.2
opencv-python-headless>=4.8,<5.0
Pillow>=10.0
```

若 final model 使用 MobileNetV3-Small，還需要：

```txt
torchvision>=0.17
```

安裝 training / compression dependencies：

```powershell
pip install -r requirements-train.txt
```

`requirements-train.txt` 應包含：

```txt
-r requirements.txt

mediapipe>=0.10.14,<0.11.0
tqdm>=4.66
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
albumentations>=1.4,<2.0
```

---

## Scoring

總分 120。

| Criteria | Points | Rule |
|---|---:|---|
| Model Size | 30 | `(10 - size_MB) × 3`，超過 10 MB 得 0 |
| Basic Performance | 20 | HaGRIDv2-based test set |
| Real-World Robustness | 40 | TA-shot real-world set |
| Presentation | 30 | live demo + defense mechanism explanation |

逐筆 scoring：

| GT | Prediction | Score |
|---|---|---:|
| target class | correct target class | +1 |
| target class | wrong class / N/A | -2 |
| N/A | any target class | -2 |
| N/A | N/A | 0 |

設計重點：

```txt
false trigger 很貴
N/A handling 很重要
plain accuracy 不夠
confidence threshold 需要調
landmark-based rejection 可以提升 robustness
```

---

## References

```txt
Han et al.,
Deep Compression: Compressing Deep Neural Networks with Pruning,
Trained Quantization and Huffman Coding,
ICLR 2016.
```

作為 compression baseline 參考。`baseline.py` 對應：

```txt
pruning
trained quantization / weight sharing
Huffman coding
```

```txt
Howard et al.,
Searching for MobileNetV3,
ICCV 2019.
```

作為 image backbone 參考。

```txt
Zhang et al.,
MediaPipe Hands: On-device Real-time Hand Tracking,
2020.
```

作為 21-point landmark representation 參考。

```txt
Nuzhdin et al.,
HaGRIDv2: 1M Images for Static and Dynamic Hand Gesture Recognition,
2024.
```

作為 dataset 參考。

HaGRID / HaGRIDv2 gesture recognition pretrained weights 不可用。

允許：

```txt
ImageNet-pretrained backbone
MediaPipe Hand Landmarker
```

---

## Submission

預期 zip：

```txt
team_X.zip
├── inference.py
├── model/
│   └── gesture_model.ptmodel
├── src/
├── requirements.txt
└── README.md
```

Windows 建立 zip：

```powershell
Compress-Archive -Path inference.py,model,src,requirements.txt,README.md -DestinationPath team_X.zip
```

不要放：

```txt
data/
checkpoints/
requirements-train.txt
```

`hand_preprocess.py` 由 TA 提供，除非最後規定改變，否則不需要放入 submission。