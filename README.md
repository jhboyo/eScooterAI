# Safety Vision AI

딥러닝 기반 건설현장 안전 장비(PPE) 착용 모니터링 플랫폼

## 팀 정보

- **팀명**: Safety Vision AI
- **프로젝트**: 딥러닝 기반 건설현장 안전 장비 착용 모니터링 플랫폼
- **멤버**: 김상진, 김준호, 김한솔, 유승근, 홍준재

---

## 연구 배경


### 산업안전보건기준에 관한 규칙 제32조 제1항

> 건설 사업주는 낙하·충돌 등 위험이 있는 작업에서 근로자에게 **안전모·안전조끼 등 보호구를 지급하고 착용**하도록 해야 함.

### 현황 및 문제점

- **현재 방식**: 인력 중심의 수동 점검
- **한계점**:
  - 대형 공사 현장에서 모든 작업자 실시간 감시 불가능
  - 인적 오류 및 피로도로 인한 누락 발생
  - 안전사고 대응 지연

### 솔루션

**AI 기반 자동 안전 모니터링 시스템**을 통해:
- 실시간 PPE 착용 상태 자동 감지

  <img src="materials/train_batch0.jpg" width="600" alt="Train Batch Sample">
- (미착용 시 즉각적인 알림)
- (24시간 지속적인 모니터링 가능)

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | 산업현장의 작업자 개인보호구(PPE) 착용 상태 감지 |
| **탐지 대상** | 헬멧(helmet), 안전조끼(vest) |
| **모델** | YOLOv8 (Transfer Learning) |
| **데이터셋** | 15,081장 (Kaggle 2개 데이터셋 통합) |

## 빠른 시작

### 환경 설정
```bash
# 의존성 설치
uv sync

# 환경 변수 파일 생성 (최초 1회)
cp .env.example .env

# .env 파일에서 PROJECT_ROOT를 본인의 경로로 수정
# 예: PROJECT_ROOT=/Users/username/workspace/SafetyVisionAI

# 가상환경 활성화 (자동 관리)
source .venv/bin/activate
```

### 데이터 전처리
```bash
# 전체 전처리 실행
uv run python src/1_preprocess/preprocess_all.py

# 또는 단계별 실행
uv run python src/1_preprocess/step1_convert_voc_to_yolo.py
uv run python src/1_preprocess/step2_verify_dataset2.py
uv run python src/1_preprocess/step3_merge_datasets.py
uv run python src/1_preprocess/step4_split_dataset.py
uv run python src/1_preprocess/step5_generate_yaml.py
uv run python src/1_preprocess/step6_validate_dataset.py
```

### 모델 훈련
```bash
uv run python src/2_training/train.py --data configs/ppe_dataset.yaml
```

### 추론
```bash
# 이미지 추론
uv run python src/3_inference/inference.py --model models/best_model.pt --input test_image.jpg

# 웹캠 실시간 추론
uv run python src/inference.py --model models/best_model.pt --source webcam
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **언어** | Python 3.11 |
| **패키지 관리** | uv |
| **딥러닝** | PyTorch, Ultralytics (YOLOv8) |
| **이미지 처리** | OpenCV, NumPy |
| **시각화** | Matplotlib |
| **웹 UI** | Streamlit (예정) |

---

## 프로젝트 구조

```
SafetyVisionAI/
├── configs/                # 설정 파일
│   ├── ppe_dataset.yaml   # 데이터셋 설정 (YOLO 필수)
│   └── train_config.yaml  # 훈련 하이퍼파라미터
├── dataset/                # 데이터셋
│   ├── data/              # 훈련 데이터 (Hugging Face 업로드용)
│   │   ├── train/         # 훈련 데이터 (70%)
│   │   ├── val/           # 검증 데이터 (15%)
│   │   └── test/          # 테스트 데이터 (15%)
│   └── raw_data/          # 전처리 전 원본
│       ├── raw/           # 원본 데이터
│       └── processed/     # 전처리 중간 결과
├── models/                 # 훈련된 모델
├── src/                    # 소스 코드
│   ├── 1_preprocess/      # 전처리 스크립트
│   ├── 2_training/        # 훈련 스크립트
│   ├── 3_inference/       # 추론 스크립트
│   └── 4_test/            # 테스트 스크립트
├── notebooks/              # Jupyter 노트북
│   └── preprocess/        # 전처리 노트북
├── materials/              # 참고 자료
├── pyproject.toml          # 의존성 정의
└── README.md
```

---

## 진행 현황

### Phase 1: 환경 설정 ✅
- [x] Python 가상환경 생성 (uv)
- [x] 라이브러리 설치
- [x] 프로젝트 구조 생성

### Phase 2: Dataset 준비 & 전처리 진행 ✅
- [x] Step 1: Dataset 1 VOC → YOLO 변환 (4,581개)
- [x] Step 2: Dataset 2 클래스 ID 확인 (10,500개)
- [x] Step 3: Dataset 통합 (15,081개)
- [x] Step 4: Train/Val/Test 분할 (70/15/15)
- [x] Step 5: Dataset YAML 생성
- [x] Step 6: 데이터 검증 및 시각화

### Phase 3: 모델 훈련 🔄 (진행 중)
- [x] YOLOv8 모델 선택 (yolov8n - Nano)
- [x] 훈련 설정 파일 작성
- [x] 클래스 정의 (helmet, vest)
- [x] Transfer Learning 실행 (MacBook 3 epochs 테스트 완료)
- [ ] 전체 훈련 (100 epochs) - RunPod A100 또는 Colab T4 예정
- [ ] 하이퍼파라미터 튜닝

### Phase 4: 모델 평가 ⏳
- [ ] `src/evaluate.py` 작성
- [ ] mAP@0.5, mAP@0.5:0.95 측정
- [ ] Precision, Recall, F1-Score 계산
- [ ] 클래스별 성능 분석
- [ ] Confusion Matrix 생성
- [ ] FPS 측정

### Phase 5: 추론 시스템 ⏳
- [ ] `src/inference.py` 작성
- [ ] 이미지 추론
- [ ] 비디오 파일 추론(예정)
- [ ] 웹캠 실시간 추론(예정)
- [ ] 결과 시각화 (바운딩 박스, 클래스명, 신뢰도)

### Phase 6: 웹 인터페이스 ⏳
- [ ] Streamlit 대시보드
- [ ] 실시간 모니터링

---

## 설정 파일

### ppe_dataset.yaml
YOLO 모델이 데이터를 찾기 위한 **필수** 설정 파일

```yaml
path: /path/to/project/images   # 절대 경로 (자동 생성)
train: train/images
val: val/images
test: test/images

nc: 2
names:
  0: helmet
  1: vest
```

**주의:** 이 파일의 `path`는 `.env`의 `PROJECT_ROOT`를 기반으로 자동 생성됩니다.

**다른 개발자 설정 방법:**
```bash
# 1. .env 파일에서 PROJECT_ROOT를 본인의 경로로 수정
# 예: PROJECT_ROOT=/Users/username/workspace/SafetyVisionAI

# 2. YAML 파일 재생성
uv run python src/1_preprocess/step5_generate_yaml.py
```

**사용:**
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='configs/ppe_dataset.yaml', epochs=100)
```

### train_config.yaml
훈련 하이퍼파라미터 관리 파일

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| epochs | 100 | 학습 반복 횟수 |
| batch_size | 16 | 배치 크기 (GPU 메모리에 따라 조절) |
| lr0 | 0.01 | 초기 학습률 |
| img_size | 640 | 입력 이미지 크기 |

### 훈련 출력 파일
훈련 완료 후 `models/ppe_detection/` 폴더에 생성되는 파일들:

| 분류 | 파일 | 설명 |
|------|------|------|
| **핵심 결과** | `weights/best.pt` | 최고 성능 모델 |
| **핵심 결과** | `weights/last.pt` | 마지막 체크포인트 |
| **성능 지표** | `results.csv` | 에포크별 훈련 통계 |
| **평가 시각화** | `confusion_matrix.png` | 혼동 행렬 |
| **평가 시각화** | `PR_curve.png` | Precision-Recall 곡선 |
| **평가 시각화** | `training_curves.png` | 훈련 결과 그래프 (visualize_results.py로 생성) |
| **디버깅용** | `train_batch*.jpg` | 데이터 로딩/augmentation 확인용 |
| **디버깅용** | `labels.jpg` | 클래스별 객체 수 분포 |
| **디버깅용** | `args.yaml` | 훈련에 사용된 설정값 |

**참고:** 디버깅용 파일들(`train_batch*.jpg`, `labels.jpg`, `args.yaml`)은 훈련 시작 시 자동 생성되며, `.gitignore`에 포함되어 Git에서 추적되지 않습니다.

<img src="models/ppe_detection/training_curves.png" width="600" alt="Training Curves">

---

## 데이터셋

### 데이터 출처

| Dataset | 이미지 수 | 원본 형식 | 클래스 |
|---------|-----------|-----------|--------|
| Hard Hat Detection | 5,000 | Pascal VOC | helmet, head, person |
| Safety Helmet & Jacket | 10,500 | YOLO | helmet, vest |

### 클래스 매핑

| 통일 클래스 | Dataset 1 | Dataset 2 |
|-------------|-----------|-----------|
| 0: helmet | helmet | Safety-Helmet |
| 1: vest | - | Reflective-Jacket |
| (제외) | head, person | - |

### 최종 데이터셋

| 구분 | 이미지 수 | 비율 |
|------|-----------|------|
| Train | 10,556 | 70% |
| Val | 2,262 | 15% |
| Test | 2,263 | 15% |
| **합계** | **15,081** | 100% |

---

## 전처리 상세

### Step 1: VOC → YOLO 변환
Dataset 1 (Hard Hat Detection)을 Pascal VOC에서 YOLO 형식으로 변환

```python
# 클래스 매핑
dataset1_mapping = {
    'helmet': 0,   # 사용
    'head': -1,    # 제외
    'person': -1   # 제외
}
```

**결과:** 5,000개 → 4,581개 (helmet이 있는 이미지만)

### Step 2: Dataset 2 확인
Dataset 2는 이미 YOLO 형식이므로 클래스 ID만 확인

**결과:** 10,500개 (helmet: 20,191개, vest: 16,049개)

### Step 3: 데이터 통합
두 데이터셋을 prefix로 구분하여 병합

- `ds1_` : Dataset 1 파일
- `ds2_` : Dataset 2 파일

**결과:** 15,081개

### Step 4: Train/Val/Test 분할
70/15/15 비율로 랜덤 분할

### Step 5: YAML 생성
`configs/ppe_dataset.yaml` 생성 (상대 경로 사용)

### Step 6: 데이터 검증
이미지-라벨 매칭 확인 및 시각화

**검증 결과:**
| 구분 | 이미지 | 라벨 | 매칭 | helmet | vest |
|------|--------|------|------|--------|------|
| Train | 10,556 | 10,556 | 100% | 27,240 | 11,334 |
| Val | 2,262 | 2,262 | 100% | 5,973 | 2,279 |
| Test | 2,263 | 2,263 | 100% | 5,944 | 2,436 |
| **합계** | **15,081** | **15,081** | **100%** | **39,157** | **16,049** |

- 모든 이미지-라벨 매칭 완료 (누락 없음)
- 샘플 이미지: `dataset/raw_data/processed/samples/`

---

## 모델 훈련 상세

### Step 1: 환경 설정
`.env` 파일에서 프로젝트 경로 설정

```bash
# .env 파일 생성
cp .env.example .env

# PROJECT_ROOT를 본인의 경로로 수정
# 예: PROJECT_ROOT=/Users/username/workspace/SafetyVisionAI
```

### Step 2: 데이터셋 다운로드 (Hugging Face)
Hugging Face에서 데이터셋 다운로드

```bash
# hf CLI 설치
uv tool install huggingface-hub

# 로그인 (최초 1회)
uv tool run hf auth login

# 데이터셋 다운로드
uv tool run hf download jhboyo/ppe-dataset --repo-type dataset --local-dir ./dataset/data
```

### Step 3: 데이터셋 YAML 생성
`.env`의 `PROJECT_ROOT`를 기반으로 데이터셋 설정 파일 생성

```bash
uv run python src/1_preprocess/step5_generate_yaml.py
```

**결과:** `configs/ppe_dataset.yaml` 생성 (절대 경로 포함)

### Step 4: 훈련 실행
YOLOv8 모델 훈련 (Transfer Learning)

```bash
# 기본 실행 (configs/train_config.yaml 사용)
uv run python src/2_training/train.py

# 데이터셋 직접 지정
uv run python src/2_training/train.py --data configs/ppe_dataset.yaml
```

**훈련 설정:**
- 모델: YOLOv8n (Nano) - COCO 사전학습 가중치
- Epochs: 100
- Batch Size: 16
- Image Size: 640x640
- Optimizer: AdamW

### Step 5: 결과 시각화
훈련 결과를 그래프로 시각화

```bash
uv run python src/2_training/visualize_results.py
```

**결과:** `models/ppe_detection/training_curves.png` 생성

### Step 6: 모델 평가 (예정)
테스트 데이터셋으로 모델 성능 평가

```bash
uv run python src/4_test/evaluate.py
```

---

## 성능 목표

| 지표 | 목표값 |
|------|--------|
| mAP@0.5 | > 85% |
| FPS | > 30 (실시간) |

---

## 일정

| 주차 | 기간 | 목표 |
|------|------|------|
| Week 1 | ~11/24 | 데이터셋 확보 및 전처리 완료 |
| Week 2 | ~12/1 | 모델 훈련 및 추론 시스템 완료 |
| Week 3 | ~12/7 | 최종 시스템 완성 및 발표 준비 |

---

## 참고 자료

- 논문: `딥 러닝 기반 작업자 개인보호구 착용 및 얼굴 신원 확인 시스템에 관한 연구`
- 논문: `Construction Site Hazards Identification Using Deep Learning and Computer Vision`
- 논문: `YOLO(You Only Look Once) 모델별 건설 현장 위험 상태 및 객체 인식 성능 비교`
- 특허: `빅데이터 기술 및 인공지능 기술에 기초하여 위험 시설물에 대한 실시간 정보를 모니터링함과 함께 상기 위험 시설물의 안전사고를 관리하는 위험 시설물 관리 시스템`
- 특허: `인공지능기반 이륜자동차의 헬맷 미착용 단속시스템 및 방법`
- 논문: `SYSTEM AND METHOD FOR AI VISUAL INSPECTION`
- [YOLO 공식 문서](https://docs.ultralytics.com/)
- [영상 - 중대재해법 비웃는 건설현장](https://www.youtube.com/watch?v=9rDv59u3cnc)
- [스타트업 미스릴 브로셔](https://6542f7fa-15be-45d4-980e-46706516dc78.usrfiles.com/ugd/6542f7_9f7aaea5869742518907c1a3bf09ba8a.pdf)


