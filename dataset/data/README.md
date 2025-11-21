---
license: mit
task_categories:
  - object-detection
language:
  - en
tags:
  - yolo
  - ppe-detection
  - safety
  - computer-vision
  - construction
size_categories:
  - 10K<n<100K
---

# PPE Detection Dataset (3-Class)

딥러닝 기반 건설현장 안전 장비(PPE) 착용 모니터링을 위한 데이터셋

## Dataset Description

개인보호구(Personal Protective Equipment) **착용/미착용 상태**를 감지하기 위한 YOLO 형식의 객체 탐지 데이터셋입니다.

**주요 특징:**
- ✅ 헬멧 착용 감지 (helmet)
- ⚠️ **헬멧 미착용 감지 (head)** - 실시간 안전 경고 가능
- ✅ 안전조끼 착용 감지 (vest)
- 15,500개 이미지, 60,991개 객체
- YOLOv8 최적화 포맷

### Classes

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | helmet | 안전 헬멧 착용 ✅ |
| 1 | head | 헬멧 미착용 (머리만) ⚠️ |
| 2 | vest | 반사 안전 조끼 착용 ✅ |

### Dataset Statistics

| Split | Images | Labels | Helmet | Head | Vest | Total Objects |
|-------|--------|--------|--------|------|------|---------------|
| Train | 9,999 | 9,999 | 25,425 | 3,679 | 10,351 | 39,455 |
| Val | 2,750 | 2,750 | 6,793 | 1,144 | 2,737 | 10,674 |
| Test | 2,751 | 2,751 | 6,939 | 962 | 2,961 | 10,862 |
| **Total** | **15,500** | **15,500** | **39,157** | **5,785** | **16,049** | **60,991** |

**Class Distribution:**
- Helmet: 39,157개 (64.2%) - 헬멧 착용
- Head: 5,785개 (9.5%) - 헬멧 미착용
- Vest: 16,049개 (26.3%) - 안전조끼 착용

**Split Ratio:**
- Train: 64.5% (9,999 images)
- Val: 17.7% (2,750 images)
- Test: 17.7% (2,751 images)

### Data Format

YOLO 형식 (normalized coordinates):
```
class_id x_center y_center width height
```

Example:
```
0 0.456789 0.345678 0.123456 0.234567  # helmet
1 0.234567 0.123456 0.098765 0.187654  # head
2 0.567890 0.456789 0.145678 0.256789  # vest
```

## Dataset Structure

```
ppe-dataset/
├── train/
│   ├── images/     # 9,999 images
│   └── labels/     # 9,999 label files (3 classes)
├── val/
│   ├── images/     # 2,750 images
│   └── labels/     # 2,750 label files (3 classes)
└── test/
    ├── images/     # 2,751 images
    └── labels/     # 2,751 label files (3 classes)
```

## Usage

### Download with Hugging Face CLI

```bash
# Install huggingface-hub
pip install huggingface-hub

# Download dataset
huggingface-cli download jhboyo/ppe-dataset --repo-type dataset --local-dir ./dataset
```

### Using with uv

```bash
uv tool install huggingface-hub
uv tool run hf download jhboyo/ppe-dataset --repo-type dataset --local-dir ./dataset/data
```

### YOLO Training Configuration

Create a YAML configuration file:

```yaml
# ppe_dataset.yaml
path: /path/to/dataset
train: train/images
val: val/images
test: test/images

nc: 3
names:
  0: helmet
  1: head
  2: vest
```

### Training with YOLOv8

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train
model.train(
    data='ppe_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

## Data Sources

This dataset is merged from two Kaggle datasets:

1. **Hard Hat Detection** (5,000 images)
   - Original classes: helmet, head, person
   - Used: **helmet, head** (착용/미착용 모두 탐지)

2. **Safety Helmet and Reflective Jacket** (10,500 images)
   - Classes: Safety-Helmet, Reflective-Jacket
   - Used: both classes (helmet, vest)

### Preprocessing

- VOC to YOLO format conversion for Dataset 1
- **3-Class Mapping:**
  - helmet: 0 (헬멧 착용)
  - head: 1 (헬멧 미착용, Dataset 1 only)
  - vest: 2 (안전조끼 착용)
- File naming with prefix (ds1_, ds2_) to avoid conflicts
- **Dataset split:**
  - Train: 64.5% (9,999 images)
  - Val: 17.7% (2,750 images)
  - Test: 17.7% (2,751 images)
  - Seed: 42 (reproducible)

## License

MIT License

## Citation

```bibtex
@dataset{ppe_detection_2024,
  title={PPE Detection Dataset for Construction Safety},
  author={SafetyVisionAI Team},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/jhboyo/ppe-dataset}
}
```

## Project

This dataset is part of the **Safety Vision AI** project - a deep learning-based construction site safety equipment monitoring platform.

## Original Dataset Sources

This dataset is created by merging and preprocessing the following Kaggle datasets:

1. **Hard Hat Detection Dataset**
   - Source: [Hard Hat Detection on Kaggle](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
   - Original classes: helmet, head, person
   - Format: Pascal VOC
   - Images: 5,000

2. **Safety Helmet and Reflective Jacket Dataset**
   - Source: [Construction Site Safety Image Dataset on Kaggle](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow)
   - Original classes: Safety-Helmet, Reflective-Jacket
   - Format: YOLO
   - Images: 10,500

**Acknowledgments:** We thank the original dataset creators for making their work publicly available.

