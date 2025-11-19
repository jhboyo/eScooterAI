# Safety Vision AI - 딥러닝 기반 건설현장 안전 장비 착용 모니터링 플랫폼

딥러닝 기반의 작업자 개인보호구(PPE) 착용 감지 및 산업안전 재해 방지를 위한 머신러닝 모델 개발 프로젝트

## 📋 프로젝트 진행 순서

### Phase 1: 환경 설정 ✅ 완료
- [x] Python 가상환경 생성 (uv)
- [x] TensorFlow, OpenCV, Jupyter 등 라이브러리 설치
- [x] 프로젝트 폴더 구조 생성

### Phase 2: 데이터셋 준비 🔄 진행 중
1. **데이터셋 수집** ✅ 완료
   - [x] Hard Hat Detection (Kaggle) 다운로드
   - [x] Safety Helmet and Reflective Jacket (Kaggle) 다운로드

2. **데이터 전처리** ⏳ 대기
   - [ ] 라벨링 포맷 통일 (Pascal VOC XML → YOLO TXT)
   - [ ] 클래스 ID 매핑 통일 (helmet, head/no_helmet, vest 등)
   - [ ] 이미지-라벨 파일 매칭 검증
   - [ ] 손상된 파일 및 잘못된 라벨 제거

3. **데이터 분할** ⏳ 대기
   - [ ] Train/Val/Test 분할 (70/15/15)
   - [ ] 데이터셋 YAML 파일 작성 (configs/ppe_dataset.yaml)

4. **데이터 검증** ⏳ 대기
   - [ ] 라벨 시각화로 정확성 확인
   - [ ] 클래스 분포 분석

### Phase 3: 모델 훈련 ⏳ 대기
1. **모델 설정**
   - [ ] YOLOv8 모델 선택 (yolov8n 또는 yolov8s)
   - [ ] 훈련 설정 파일 작성 (configs/train_config.yaml)
   - [ ] 클래스 정의 (helmet, head, vest)

2. **Transfer Learning**
   - [ ] COCO 사전 훈련 가중치 로드
   - [ ] PPE 데이터셋으로 Fine-tuning (50-100 epochs)
   - [ ] 훈련 진행 모니터링 (loss, mAP)

3. **하이퍼파라미터 튜닝**
   - [ ] 학습률 조정 (0.01, 0.001, 0.0001)
   - [ ] 배치 크기 조정 (8, 16, 32)
   - [ ] 데이터 증강 설정 (mosaic, flip, hsv 등)

### Phase 4: 모델 평가 ⏳ 대기
1. **성능 평가**
   - [ ] mAP@0.5, mAP@0.5:0.95 측정
   - [ ] Precision, Recall, F1-Score 계산
   - [ ] 클래스별 성능 분석
   - [ ] FPS 측정 (실시간 처리 성능)

2. **모델 개선**
   - [ ] 오탐지(False Positive) 분석
   - [ ] 미탐지(False Negative) 분석
   - [ ] Confusion Matrix 분석
   - [ ] 추가 훈련 또는 파라미터 조정

### Phase 5: 추론 시스템 구현 ⏳ 대기
1. **추론 스크립트 작성**
   - [ ] 이미지 추론 (src/inference.py)
   - [ ] 비디오 파일 추론
   - [ ] 웹캠 실시간 추론

2. **결과 시각화**
   - [ ] 바운딩 박스 표시
   - [ ] 클래스명, 신뢰도 표시
   - [ ] 결과 이미지/비디오 저장

### Phase 6: 웹 인터페이스 (선택) ⏳ 대기
- [ ] Streamlit 기반 대시보드 구현
- [ ] 실시간 모니터링 화면
- [ ] 이미지/비디오 업로드 기능
- [ ] 결과 저장 및 리포트 생성

---

## 📊 데이터셋 분석

### Dataset 1: kaggle-safey_helmet (Hard Hat Detection)
| 항목 | 내용 |
|------|------|
| **이미지 수** | 5,000장 |
| **라벨 형식** | Pascal VOC (XML) |
| **이미지 형식** | PNG (416x416) |
| **클래스** | helmet (18,966개), head (5,785개), person (751개) |

### Dataset 2: safety-Helmet-Reflective-Jacket
| 항목 | 내용 |
|------|------|
| **이미지 수** | Train 7,350 / Valid 1,575 / Test 1,575 (총 10,500장) |
| **라벨 형식** | YOLO (TXT) ✅ |
| **이미지 형식** | JPG |
| **클래스** | 0: Safety-Helmet (10,868개), 1: Reflective-Jacket (8,210개) |

### 클래스 매핑 계획
| 통일 클래스 | Dataset 1 원본 | Dataset 2 원본 |
|-------------|----------------|----------------|
| 0: helmet | helmet | 0: Safety-Helmet |
| 1: vest | ❌ 없음 | 1: Reflective-Jacket |
| - | head (제외) | - |
| - | person (제외) | - |

### 최종 데이터셋 예상
| 클래스 | Dataset 1 | Dataset 2 | 총 개수 |
|--------|-----------|-----------|---------|
| helmet | 18,966 | 10,868 | ~29,834 |
| vest | 0 | 8,210 | ~8,210 |

**총 이미지**: ~15,500장 | **총 객체**: ~38,044개

---

## 🔧 전처리 실행 (Phase 2-2)

### 실행 방법

#### 전체 실행 (한 번에)
```bash
uv run python src/preprocess_all.py
```

#### 단계별 실행
```bash
# Step 1: Dataset 1 VOC → YOLO 변환
uv run python src/preprocess/step1_convert_voc_to_yolo.py

# Step 2: Dataset 2 클래스 ID 확인
uv run python src/preprocess/step2_verify_dataset2.py

# Step 3: 데이터셋 통합
uv run python src/preprocess/step3_merge_datasets.py

# Step 4: Train/Val/Test 분할
uv run python src/preprocess/step4_split_dataset.py

# Step 5: YAML 파일 생성
uv run python src/preprocess/step5_generate_yaml.py

# Step 6: 데이터 검증
uv run python src/preprocess/step6_validate_dataset.py
```

---

### Step 1: Dataset 1 변환 (VOC → YOLO) ✅ 완료
```python
# 클래스 매핑
dataset1_mapping = {
    'helmet': 0,   # → helmet
    'head': -1,    # → 제외
    'person': -1   # → 제외
}
```

**실행 결과:**
- 입력: 5,000개 XML 파일
- 변환됨: 4,581개 (helmet이 있는 이미지)
- 스킵됨: 419개 (helmet 없음)
- 출력: `images/processed/dataset1/`

---

### Step 2: Dataset 2 클래스 ID 확인 ✅ 완료
```python
# 클래스 매핑 (이미 YOLO 형식)
dataset2_mapping = {
    0: 0,  # Safety-Helmet → helmet
    1: 1   # Reflective-Jacket → vest
}
```

**실행 결과:**
- 총 이미지: 10,500개 (Train 7,350 / Valid 1,575 / Test 1,575)
- helmet: 20,191개
- vest: 16,049개
- 결론: 변환 불필요, 그대로 사용 가능

---

### Step 3: 데이터 통합 ⏳ 대기
- 두 데이터셋을 `images/processed/merged/`로 병합
- 파일명 충돌 방지 (prefix 추가: `ds1_`, `ds2_`)

---

### Step 4: Train/Val/Test 분할 ⏳ 대기
- Train/Val/Test 재분할 (70/15/15)
- `images/train/`, `images/val/`, `images/test/`에 저장

---

### Step 5: 데이터셋 YAML 생성 ⏳ 대기
```yaml
# configs/ppe_dataset.yaml
path: /path/to/SafetyVisionAI/images
train: train/images
val: val/images
test: test/images

nc: 2
names:
  0: helmet
  1: vest
```

---

### Step 6: 데이터 검증 ⏳ 대기
- 이미지-라벨 매칭 확인
- 클래스 분포 시각화
- 샘플 이미지 바운딩박스 시각화

---

## ⚡ 집중 개발 전략 (3주 단축 계획)

### 핵심 기능 우선순위
1. **필수 기능**: PPE 탐지 (헬멧, 안전조끼)
2. **추가 기능**: 웹 인터페이스 모니터링
3. **선택 기능**: 고급 분석, 리포트 생성

### 시간 단축 방법
- **사전 훈련된 모델 활용**: 처음부터 훈련하지 않고 Transfer Learning 사용
- **공개 데이터셋 활용**: 자체 데이터 수집 대신 검증된 공개 데이터셋 사용
- **경량 모델 선택**: MobileNet 기반으로 빠른 개발과 추론
- **최소 기능 구현**: 핵심 기능에 집중, 부가 기능 최소화

## 🛠️ 기술 스택

### 머신러닝/딥러닝
- **Framework**: TensorFlow 2.13.0 ✅
- **모델**: MobileNet-SSD, EfficientDet (경량화 우선)
- **전처리**: OpenCV, NumPy
- **시각화**: Matplotlib

### 개발 환경
- **언어**: Python 3.11 ✅
- **가상환경**: uv ✅
- **버전관리**: Git
- **노트북**: Jupyter ✅

### 배포 및 서빙
- **웹 인터페이스**: Streamlit (빠른 구현)
- **추론 최적화**: TensorFlow Lite (모바일 최적화)
- **실시간 처리**: OpenCV VideoCapture

## 📁 프로젝트 구조

```
SafetyVisionAI/
├── materials/              # 프로젝트 관련 문서 및 자료
│   ├── papers/            # 연구논문
│   ├── patents/           # 특허 자료
│   └── company/           # 회사 자료
├── data/                   # 데이터셋
│   ├── raw/               # 원본 데이터 (다운로드한 그대로)
│   ├── processed/         # 전처리된 데이터
│   ├── train/             # 훈련용 데이터
│   │   ├── images/       # 이미지
│   │   └── labels/       # 라벨 (YOLO/COCO/VOC 형식)
│   ├── val/               # 검증용 데이터
│   │   ├── images/
│   │   └── labels/
│   └── test/              # 테스트용 데이터
│       ├── images/
│       └── labels/
├── models/                # 훈련된 모델 파일
│   ├── best_model.pt      # 최고 성능 모델
│   ├── last_model.pt      # 마지막 체크포인트
│   └── checkpoints/       # 중간 체크포인트들
├── src/                   # 소스 코드
│   ├── preprocess.py      # 데이터 전처리
│   ├── dataloader.py      # 데이터 로더
│   ├── train.py           # 모델 훈련
│   ├── evaluate.py        # 모델 평가
│   ├── inference.py       # 추론/예측
│   └── utils.py           # 유틸리티 함수
├── notebooks/             # Jupyter 노트북
│   ├── data_analysis.ipynb    # 데이터 분석
│   ├── model_training.ipynb   # 모델 훈련 실험
│   └── visualization.ipynb    # 결과 시각화
├── configs/               # 설정 파일
│   ├── train_config.yaml  # 훈련 설정
│   └── model_config.yaml  # 모델 설정
├── pyproject.toml         # Python 의존성 (uv 사용)
├── uv.lock               # 의존성 락파일
├── main.py               # 메인 실행 파일
├── CLAUDE.md             # 프로젝트 지침서
└── README.md             # 프로젝트 설명
```

## 🎯 주요 마일스톤 (12월 7일 마감)

- **11/24 (Week 1)**: 데이터셋 확보 및 기본 모델 구현 완료
- **12/1 (Week 2)**: 모델 훈련 및 추론 시스템 완료
- **12/7 (Week 3)**: 최종 시스템 완성 및 발표 준비 완료

## ⏰ 일정 관리 전략

### 위험 요소 및 대응 방안
1. **데이터셋 확보 지연** → 다중 소스에서 동시 다운로드
2. **모델 훈련 시간 부족** → 클라우드 GPU 활용 검토
3. **통합 테스트 시간 부족** → 주간별 점검 강화

### 효율성 극대화 방법
- **병렬 작업**: 데이터 전처리와 모델 연구 동시 진행
- **일일 체크포인트**: 매일 진행상황 점검 및 조정
- **MVP 접근법**: 최소 기능 제품 먼저 완성 후 개선

## 🔄 개발 워크플로우 (데이터셋 수집 후)

### 1. 데이터 전처리 (Preprocessing)
```bash
uv run python src/preprocess.py --input data/raw/ --output data/processed/
```
- **데이터 검증**: 이미지와 라벨 파일 매칭 확인
- **이미지 리사이징**: 모델 입력 크기 맞춤 (예: 640x640)
- **정규화**: 픽셀 값 0-1 또는 -1~1로 변환
- **라벨 포맷 변환**: COCO → YOLO, Pascal VOC 등
- **데이터 정제**: 손상된 파일, 잘못된 라벨 제거

### 2. 데이터 분할 (Train/Validation/Test Split)
```
데이터셋 분할 비율:
├── train (70-80%): 모델 학습용
├── val (10-15%): 하이퍼파라미터 튜닝, 조기 종료
└── test (10-15%): 최종 성능 평가
```

### 3. 데이터 증강 (Data Augmentation)
훈련 데이터 다양성 증가:
- 회전 (Rotation)
- 반전 (Horizontal/Vertical Flip)
- 밝기/대비 조절 (Brightness/Contrast)
- 노이즈 추가 (Gaussian Noise)
- Mosaic, Mixup
- Random Crop/Scale

### 4. 모델 선택 및 설정
```yaml
# configs/train_config.yaml
model: yolov8n        # yolov5, yolov8, faster-rcnn 등
num_classes: 7        # PPE 클래스 수
input_size: 640       # 입력 이미지 크기
batch_size: 16
epochs: 100
learning_rate: 0.001
```

**모델 옵션:**
- **YOLO (v5, v8, v10)**: 빠른 실시간 탐지 (권장)
- **Faster R-CNN**: 높은 정확도, 느린 속도
- **EfficientDet**: 정확도-속도 균형
- **DETR**: Transformer 기반

### 5. 모델 훈련 (Training)
```bash
uv run python src/train.py --config configs/train_config.yaml
```
- **전이 학습** (Transfer Learning): COCO 사전 훈련 가중치 사용
- **손실 함수**: Classification Loss + Localization Loss
- **옵티마이저**: Adam, SGD, AdamW 등
- **학습률 스케줄링**: CosineAnnealing, StepLR
- **조기 종료** (Early Stopping): 과적합 방지
- **체크포인트 저장**: 최고/최신 모델 저장

### 6. 모델 평가 (Evaluation)
```bash
uv run python src/evaluate.py --model models/best_model.pt --data data/test/
```
**평가 지표:**
- **mAP** (mean Average Precision): @0.5, @0.75, @0.5:0.95
- **Precision**: 정밀도 (정확히 예측한 비율)
- **Recall**: 재현율 (놓치지 않은 비율)
- **F1-Score**: Precision과 Recall의 조화평균
- **FPS** (Frames Per Second): 실시간 처리 성능

### 7. 하이퍼파라미터 튜닝
최적화할 파라미터:
```yaml
learning_rate: [0.001, 0.0001, 0.00001]
batch_size: [8, 16, 32]
optimizer: [adam, sgd, adamw]
weight_decay: [0.0005, 0.001]
augmentation_strength: [weak, medium, strong]
```

### 8. 모델 추론 및 테스트
```bash
# 이미지 추론
uv run python src/inference.py --model models/best_model.pt --input test_image.jpg

# 비디오 추론
uv run python src/inference.py --model models/best_model.pt --input video.mp4

# 웹캠 실시간 추론
uv run python src/inference.py --model models/best_model.pt --source webcam
```

### 9. 모델 배포 준비
- **모델 경량화**: Pruning, Quantization (INT8)
- **포맷 변환**: ONNX, TensorRT, TensorFlow Lite
- **API 서버**: Flask, FastAPI로 REST API 구축
- **실시간 처리**: CCTV, 웹캠 연동

## 📋 라벨링 데이터 형식

### YOLO 형식 (.txt)
```
# class_id x_center y_center width height (모두 정규화된 0-1 값)
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.2
```

### COCO 형식 (.json)
```json
{
  "images": [{"id": 1, "file_name": "image001.jpg", "width": 1920, "height": 1080}],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 250],
      "area": 50000,
      "iscrowd": 0
    }
  ],
  "categories": [{"id": 1, "name": "helmet"}]
}
```

### Pascal VOC 형식 (.xml)
```xml
<annotation>
  <object>
    <name>helmet</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>300</xmax>
      <ymax>400</ymax>
    </bndbox>
  </object>
</annotation>
```

## 🎨 PPE 탐지 클래스 정의

```yaml
classes:
  0: helmet      # 헬멧 착용
  1: vest        # 안전조끼 착용
  2: head        # 헬멧 미착용 (머리만 보임)
```

**데이터셋 출처:**
- Hard Hat Detection (Kaggle): helmet, head 클래스
- Safety Helmet and Reflective Jacket (Kaggle): helmet, vest 클래스

## 🛠️ 라벨링 도구 추천

직접 라벨링이 필요한 경우:
- **LabelImg**: YOLO, Pascal VOC 형식 지원
- **CVAT**: 웹 기반, 협업 가능, 다양한 형식
- **Roboflow**: 온라인, 자동 포맷 변환, 데이터 증강
- **Labelbox**: 상용, 고급 기능
- **VGG Image Annotator (VIA)**: 오픈소스, 경량

## 📊 성공 지표

1. **정확도**: mAP@0.5 > 85%
2. **실시간성**: 30 FPS 이상 (웹캠 처리)
3. **안정성**: 24시간 연속 운영 가능
4. **사용성**: 직관적인 UI/UX

## 🔗 참고 자료

- [YOLO 공식 문서](https://docs.ultralytics.com/)
- [PyTorch 객체 탐지 튜토리얼](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [산업안전 PPE 데이터셋](https://github.com/akanametov/ppedetection)
- 프로젝트 관련 논문: `딥 러닝 기반 작업자 개인보호구 착용 및 얼굴 신원 확인 시스템에 관한 연구`

## 📞 팀 정보

- **팀명**: 3조
- **프로젝트명**: TFGuard
- **목표**: 산업현장 안전사고 예방을 위한 AI 시스템 개발