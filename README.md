# Safety Vision AI

딥러닝 기반 건설현장 안전 장비(PPE) 착용 모니터링 플랫폼

## 팀 정보

- **팀명**: Safety Vision AI
- **프로젝트**: 딥러닝 기반 건설현장 안전 장비 착용 모니터링 플랫폼
- **멤버**: 김상진, 김준호, 김한솔, 유승근, 홍준재

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | 산업현장의 작업자 개인보호구(PPE) 착용/미착용 상태 감지 |
| **탐지 대상** | 헬멧 착용(helmet), 헬멧 미착용(head), 안전조끼(vest) |
| **모델** | YOLOv8 (Transfer Learning) |
| **데이터셋** | 15,500장 (Kaggle 2개 데이터셋 통합) |

---

## 연구 배경

### <span style="color: red;">산업안전보건기준에 관한 규칙 제32조 제1항</span>

> ⚠️ 건설 사업주는 낙하·충돌 등 위험이 있는 작업에서 근로자에게 **안전모·안전조끼 등 보호구를 지급하고 착용**하도록 해야 함.

### 현황 및 문제점

- **현재 방식**: 인력 중심의 수동 점검
- **한계점**:
  - 대형 공사 현장에서 모든 작업자 실시간 감시 불가능
  - 인적 오류 및 피로도로 인한 누락 발생
  - 안전사고 대응 지연

### 솔루션

**AI 기반 자동 안전 모니터링 시스템**을 통해:
- 실시간 PPE 착용 상태 자동 감지
- **헬멧 미착용(head) 탐지로 즉각적인 경고 가능** ⚠️
- 미착용 시 즉각적인 알림 (예정)
- 24시간 지속적인 모니터링 가능 (예정)

<img src="materials/train_batch0.jpg" width="600" alt="Train Batch Sample">

---

## 훈련 결과

### ⚠️ 현재 상태: 3 Class 모델 훈련 진행 중

현재 프로젝트는 **2 Class (helmet, vest)**에서 **3 Class (helmet, head, vest)**로 전환하여 헬멧 미착용 상태를 감지할 수 있도록 개선 중입니다.

- ✅ **데이터 준비 완료**: 15,500개 이미지, 60,991개 객체 (3 class)
- ⏳ **A100 GPU 훈련 대기 중**: 100 epochs 본 훈련 예정
- 📊 **목표 성능**: mAP@0.5 ≥ 90%, Precision ≥ 88%, Recall ≥ 85%

---

## 기존 훈련 결과 (2 Class - 참고용)

### 최종 성능 지표 (A100 GPU)

| 지표 | 결과 |
|------|------|
| **mAP@0.5** | 0.9440 (94.4%) |
| **mAP@0.5:0.95** | 0.7308 (73.1%) |
| **Precision** | 0.9217 (92.2%) |
| **Recall** | 0.8911 (89.1%) |

### 클래스별 성능 분석

| 클래스 | Precision | Recall | AP@0.5 | AP@0.5:0.95 |
|--------|-----------|--------|--------|-------------|
| **Helmet** | 0.9364 (93.6%) | 0.8923 (89.2%) | 0.9444 (94.4%) | 0.6970 (69.7%) |
| **Vest** | 0.9104 (91.0%) | 0.8877 (88.8%) | 0.9434 (94.3%) | 0.7654 (76.5%) |

**분석:**
- 두 클래스 모두 **AP@0.5가 94% 이상**으로 매우 우수
- Vest가 helmet보다 **mAP@0.5:0.95에서 6.8%p 높음** → 조끼의 바운딩 박스가 더 정확
- 두 클래스 간 성능 차이가 크지 않아 **균형 잡힌 모델**

### 훈련 환경

| 항목 | 설정 |
|------|------|
| GPU | RunPod A100 |
| Epochs | 100 |
| Batch Size | 128 |
| Image Size | 640x640 |
| Model | YOLOv8n (Nano) |
| Optimizer | AdamW |
| Initial LR | 0.01 |
| AMP | True (Mixed Precision) |
| 총 훈련 시간 | 약 57분 (3,409초) |

### Loss 감소 추이

| Loss 종류 | 초기값 | 최종값 | 감소율 |
|-----------|--------|--------|--------|
| train/box_loss | 1.50 | 0.79 | 47% |
| train/cls_loss | 1.83 | 0.45 | 75% |
| train/dfl_loss | 1.38 | 0.98 | 29% |
| val/box_loss | 2.64 | 0.92 | 65% |
| val/cls_loss | 7.16 | 0.49 | 93% |
| val/dfl_loss | 3.27 | 1.03 | 69% |

모든 손실이 꾸준히 감소하며, validation loss도 함께 감소하여 **과적합(overfitting) 없이** 잘 학습되었습니다.

### 학습 곡선

<img src="models/ppe_detection/results.png" width="800" alt="Training Results">

### 혼동 행렬 (Confusion Matrix) 분석

#### 클래스별 탐지 결과

| 클래스 | 정확히 탐지 | 정확도 | 미탐율 |
|--------|-------------|--------|--------|
| **Helmet** | 5,456개 | 91% | 9% |
| **Vest** | 2,085개 | 91% | 8% |

#### 클래스 간 혼동

| 혼동 유형 | 건수 | 비율 |
|-----------|------|------|
| Helmet → Vest | 5개 | 0.08% |
| Vest → Helmet | 8개 | 0.35% |
| **총 클래스 간 혼동** | **13개** | **매우 낮음** |

#### False Positive (오탐)

| 오탐 유형 | 건수 |
|-----------|------|
| Background → Helmet | 549개 |
| Background → Vest | 287개 |

<img src="models/ppe_detection/confusion_matrix_normalized.png" width="500" alt="Confusion Matrix Normalized">

### 결과 해석

#### 강점
1. **높은 탐지 정확도**: mAP@0.5 = 94.4%로 목표(85%) 크게 초과
2. **클래스 간 혼동 최소화**: helmet-vest 혼동이 거의 없음 (13건/7,500건 이하)
3. **안정적인 학습**: 과적합 없이 꾸준한 성능 향상
4. **빠른 수렴**: 50 epoch 이후 안정화

#### 개선 가능 영역
1. **False Positive 감소**: background를 PPE로 오탐하는 경우 (836건)
2. **Recall 향상**: 일부 객체 미탐지 (helmet 9%, vest 8%)

#### 결론

이 모델은 **건설현장 PPE 탐지에 매우 적합**합니다:

- **실용성**: 91%+ 정확도로 실시간 모니터링 가능
- **신뢰성**: helmet/vest 간 혼동이 거의 없어 안전 모니터링에 신뢰할 수 있음
- **효율성**: YOLOv8n 경량 모델로 빠른 추론 속도 기대

### 결과 파일

| 파일 | 위치 |
|------|------|
| 최고 성능 모델 | `models/ppe_detection/weights/best.pt` |
| 마지막 체크포인트 | `models/ppe_detection/weights/last.pt` |
| 훈련 통계 | `models/ppe_detection/results.csv` |
| 혼동 행렬 | `models/ppe_detection/confusion_matrix.png` |
| PR 곡선 | `models/ppe_detection/BoxPR_curve.png` |

---

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

### 데이터셋 다운로드 (Hugging Face)
```bash
# hf CLI 설치
uv tool install huggingface-hub

# 로그인 (최초 1회)
uv tool run hf auth login

# 데이터셋 다운로드
uv tool run hf download jhboyo/ppe-dataset --repo-type dataset --local-dir ./dataset/data
```

### 데이터셋 YAML 생성
```bash
uv run python src/1_preprocess/step5_generate_yaml.py
```

### 모델 훈련
```bash
uv run python src/2_training/train.py --data configs/ppe_dataset.yaml
```

### 추론
```bash
# 이미지 추론
uv run python src/3_inference/inference.py --model models/ppe_detection/weights/best.pt --input test_image.jpg
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

## 데이터셋

### 데이터 출처

| Dataset | 이미지 수 | 원본 형식 | 클래스 |
|---------|-----------|-----------|--------|
| Hard Hat Detection | 5,000 | Pascal VOC | helmet, head, person |
| Safety Helmet & Jacket | 10,500 | YOLO | helmet, vest |

### 클래스 매핑 (3 Class)

| 통일 클래스 | Dataset 1 | Dataset 2 |
|-------------|-----------|-----------|
| 0: helmet | helmet | Safety-Helmet |
| 1: head | **head** ⚠️ | - |
| 2: vest | - | Reflective-Jacket |
| (제외) | person | - |

**주요 변경사항:**
- Dataset 1의 **head 클래스 포함** → 헬멧 미착용 탐지 가능
- vest 클래스 ID: 1 → 2로 변경

### 최종 데이터셋 (3 Class)

| 구분 | 이미지 수 | 비율 |
|------|-----------|------|
| Train | 9,999 | 64.5% |
| Val | 2,750 | 17.7% |
| Test | 2,751 | 17.7% |
| **합계** | **15,500** | 100% |

**분할 비율 변경 이유:**
- Train 데이터를 9,999개로 제한하여 검증/테스트 데이터 확보
- Val/Test 비율 증가로 더 신뢰성 있는 모델 평가 가능

### 데이터 검증 결과 (3 Class)

| 구분 | 이미지 | 라벨 | 매칭 | helmet | head | vest | 총 객체 |
|------|--------|------|------|--------|------|------|---------|
| Train | 9,999 | 9,999 | 100% | 25,425 | 3,679 | 10,351 | 39,455 |
| Val | 2,750 | 2,750 | 100% | 6,793 | 1,144 | 2,737 | 10,674 |
| Test | 2,751 | 2,751 | 100% | 6,939 | 962 | 2,961 | 10,862 |
| **합계** | **15,500** | **15,500** | **100%** | **39,157** | **5,785** | **16,049** | **60,991** |

**클래스 분포:**
- Helmet: 39,157개 (64.2%) - 헬멧 착용
- **Head: 5,785개 (9.5%)** - 헬멧 미착용 ⚠️
- Vest: 16,049개 (26.3%) - 안전조끼 착용

---

## 프로젝트 구조

```
SafetyVisionAI/
├── configs/                # 설정 파일
│   ├── ppe_dataset.yaml   # 데이터셋 설정 (YOLO 필수)
│   └── train_config.yaml  # 훈련 하이퍼파라미터
├── dataset/                # 데이터셋
│   ├── data/              # 훈련 데이터 (Hugging Face 업로드용, 3 class)
│   │   ├── train/         # 훈련 데이터 (64.5%, 9,999개)
│   │   ├── val/           # 검증 데이터 (17.7%, 2,750개)
│   │   └── test/          # 테스트 데이터 (17.7%, 2,751개)
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
├── materials/              # 참고 자료
├── pyproject.toml          # 의존성 정의
└── README.md
```

---

## 진행 현황

### Phase 1: 환경 설정 ✅
- [v] Python 가상환경 생성 (uv)
- [v] 라이브러리 설치
- [v] 프로젝트 구조 생성

### Phase 2: Dataset 준비 & 전처리 ✅
- [v] Step 1: Dataset 1 VOC → YOLO 변환 (4,581개, **head 클래스 포함**)
- [v] Step 2: Dataset 2 클래스 ID 확인 (10,500개)
- [v] Step 3: Dataset 통합 (15,500개, **3 class 매핑**)
- [v] Step 4: Train/Val/Test 분할 (64.5/17.7/17.7)
- [v] Step 5: Dataset YAML 생성 (nc: 3)
- [v] Step 6: 데이터 검증 및 시각화

### Phase 3: 모델 훈련 ⏳
- [v] YOLOv8 모델 선택 (yolov8n - Nano)
- [v] 훈련 설정 파일 작성 (nc: 3)
- [v] 클래스 정의 (helmet, **head**, vest)
- [v] Transfer Learning 실행 (MacBook 1 epoch 테스트 완료)
- [v] ~~전체 훈련 (100 epochs) - RunPod A100 완료~~ (2 Class 기준)
- [ ] **3 Class 본 훈련 (100 epochs) - RunPod A100 예정**
- [ ] 하이퍼파라미터 튜닝

### Phase 4: 모델 평가 ⏳
- [v] ~~mAP@0.5, mAP@0.5:0.95 측정~~ (2 Class 완료)
- [v] ~~Precision, Recall 계산~~ (2 Class 완료)
- [v] ~~Confusion Matrix 생성~~ (2 Class 완료)
- [v] ~~클래스별 성능 분석~~ (2 Class 완료)
- [ ] **3 Class 모델 성능 평가 (훈련 후)**
- [ ] **2 Class vs 3 Class 성능 비교 분석**

### Phase 5: 추론 시스템 ⏳
- [ ] 이미지 추론 (3 class 대응)
- [ ] **헬멧 미착용(head) 경고 로직 구현** ⚠️
- [ ] 비디오 파일 추론 (예정)
- [ ] 웹캠 실시간 추론 (예정)
- [ ] 결과 시각화 (바운딩 박스, 클래스명, 신뢰도, 경고 표시)

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

nc: 3
names:
  0: helmet
  1: head
  2: vest
```

**주의:** 이 파일의 `path`는 `.env`의 `PROJECT_ROOT`를 기반으로 자동 생성됩니다.

### train_config.yaml
훈련 하이퍼파라미터 관리 파일

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| **nc** | **3** | **클래스 수 (helmet, head, vest)** |
| epochs | 100 | 학습 반복 횟수 |
| batch_size | 128 | 배치 크기 (A100: 128, MacBook: 16) |
| lr0 | 0.01 | 초기 학습률 |
| img_size | 640 | 입력 이미지 크기 |

---

## 향후 과제

### 완료된 개선사항
- ✅ **헬멧 미착용 탐지**: head 클래스 추가로 헬멧 미착용 상태 감지 가능
- ✅ **데이터 재구성**: 3 class (helmet, head, vest) 데이터셋 구축 완료

### 남은 과제
1. **3 Class 모델 성능 검증**
   - A100 GPU 100 epochs 본 훈련 실행
   - 목표 성능 달성 여부 확인 (mAP@0.5 ≥ 90%)
   - 2 Class vs 3 Class 성능 비교 분석

2. **안전조끼 미착용 탐지**
   - 현재: vest 착용만 탐지
   - 개선: person 클래스 추가하여 vest 미착용자 식별
   - 구현: person 탐지 후 vest가 없으면 경고

3. **실시간 모니터링 시스템**
   - 웹캠 실시간 추론
   - 미착용 시 즉각 경고 알림
   - 통계 대시보드 (Streamlit)

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
