# Hand Gesture Classification on Edge Devices
> CS3570 Multimedia Technology — Final Project  
> Challenge from Microsoft | HaGRIDv2 · 2026

---

## Overview

A compact (≤ 10 MB) hand-gesture classifier for edge devices.  
The model takes a **cropped hand image** + **21 MediaPipe landmark coordinates**
and returns one of six class indices:

| Index | Label | Description |
|-------|-------|-------------|
| 0 | N/A | All other gestures / unknown |
| 1 | fist | Closed fist |
| 2 | like | Thumbs up |
| 3 | ok | OK sign |
| 4 | one | Index finger up |
| 5 | palm | Open palm |

---

## Environment Setup

### 1. Create conda environment (Python 3.10)

```bash
conda create -n gesture-cls python=3.10 -y
conda activate gesture-cls
```

> **Why 3.10?** Google Colab's default runtime is Python 3.10.
> MediaPipe 0.10.x has known protobuf conflicts on 3.11 and
> incomplete wheels for 3.12. Python 3.10 provides the best
> compatibility across all dependencies.

### 2. Install dependencies

**Inference only** (matches Colab submission environment):
```bash
pip install -r requirements.txt
```

**Full training environment**:
```bash
pip install -r requirements-train.txt
```

### 3. GPU support (optional, training only)

If your machine has a CUDA-capable GPU, replace the torch install with
the appropriate CUDA wheel. Example for CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Repository Structure

```
.
├── inference.py            # ← Submission entry point
├── model/
│   └── gesture_model.pth   # ← Final weights (must be ≤ 10 MB)
├── src/
│   ├── dataset.py          # HaGRIDv2 dataset loader
│   ├── model.py            # Model architecture definition
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation & confusion matrix
├── requirements.txt        # Inference-only deps (for Colab)
├── requirements-train.txt  # Full training deps
├── README.md
└── .gitignore
```

---

## Inference Interface

The submission `inference.py` implements exactly:

```python
import numpy as np

def predict(cropped_img: np.ndarray, landmarks: np.ndarray) -> int:
    """
    Args:
        cropped_img : np.ndarray, shape (H, W, 3), dtype uint8, RGB
        landmarks   : np.ndarray, shape (21, 2) or (21, 3), float32
                      — MediaPipe normalized [x, y, (z)] coordinates
    Returns:
        int in {0, 1, 2, 3, 4, 5}
        0 = N/A, 1 = fist, 2 = like, 3 = ok, 4 = one, 5 = palm
    """
    ...
    return final_decision_class
```

All file paths inside `inference.py` must be **relative to `inference.py`**.

---

## Dataset

Download HaGRIDv2 512px from the official repository:
```
https://github.com/hukenovs/hagrid/tree/Hagrid_v2-1M
```

Recommended local structure:
```
data/
├── hagridv2_512/
│   ├── fist/
│   ├── like/
│   ├── ok/
│   ├── one/
│   ├── palm/
│   └── <other_28_classes>/   # used as N/A training samples
```

> ⚠️ Do **not** commit the dataset to git. It is listed in `.gitignore`.

---

## Training

```bash
python src/train.py \
    --data_root data/hagridv2_512 \
    --epochs 30 \
    --batch_size 64 \
    --output_dir checkpoints/
```

---

## Evaluation Scoring

| Criteria | Points | Rule |
|----------|--------|------|
| Model Size | 30 pts | `(10 − size_MB) × 3`; 0 pts if > 10 MB |
| Basic Performance (HaGRIDv2) | 20 pts | +1 correct, **−2 false trigger** |
| Real-World Robustness | 40 pts | +1 correct, **−2 false trigger** (TA-shot dataset) |
| Presentation | 30 pts | Live demo + defense mechanisms explanation |

**False triggers are penalized at 2×** — conservative N/A prediction is preferred
over aggressive classification.

---

## Submission

Pack the zip file as follows:
```bash
zip -r team_X.zip inference.py model/ requirements.txt README.md
```

Submit via the provided Google Form before the deadline.  
Leaderboard updates: every Mon / Wed / Fri / Sun starting **2026-05-29**.

---

## Notes

- Pretrained weights trained on HaGRID / HaGRIDv2 are **not allowed**.  
  ImageNet-pretrained backbones (e.g. MobileNetV3-Small) are allowed.
- Engineering heuristics (confidence thresholding, landmark-based rules)
  are **explicitly encouraged** for the N/A class.
- Full-frame raw images are **strictly prohibited** as model input.
