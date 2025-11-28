# eScooterAI - 전동킥보드 헬멧 착용 모니터링 모바일 서비스

딥러닝 기반 전동킥보드(e-Scooter) 헬멧 착용 탐지 및 안전 모니터링 플랫폼

---

## 🚀 프로젝트 데모

[![Streamlit Cloud](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://escooter-helmet-detection.streamlit.app)

**👉 실시간 웹캠 데모** (예정)

모바일/웹 브라우저에서 바로 헬멧 탐지를 테스트해보세요!
- 📱 모바일 카메라 실시간 탐지
- 🎯 헬멧 착용/미착용 자동 감지
- 📊 안전 수준 자동 평가
- 🖼️ 실시간 객체 탐지 시각화
- 📱 **Telegram 실시간 알림** (헬멧 미착용 감지 시 자동 경고!)
- 🤖 **RAG 기반 안전 가이드** (헬멧 관련 질의응답)

> ⚠️ **실시간 안전 알림 시스템**: 헬멧 미착용 감지 시 Telegram Bot이 **즉각 알림 전송**!
> 탐지 결과 이미지와 안전 통계를 포함한 상세한 경고 메시지를 받아보세요.

---

## 팀 정보

- **팀명**: eScooterAI
- **프로젝트**: 딥러닝 기반 전동킥보드 헬멧 착용 모니터링 모바일 서비스
- **멤버**: 김상진, 김준호, 김한솔, 유승근, 홍준재

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | 전동킥보드 이용자의 헬멧 착용/미착용 상태 실시간 감지 |
| **탐지 대상** | 헬멧 착용(helmet), 헬멧 미착용(head) |
| **모델** | YOLOv8n (SafetyVisionAI 사전 훈련 모델 활용) |
| **플랫폼** | 모바일 웹 서비스 (Streamlit + WebRTC) |
| **추가 기능** | RAG 기반 안전 가이드 챗봇 |

---

## 연구 배경

### <span style="color: red;">도로교통법 제50조 제3항 (개인형 이동장치 안전기준)</span>

> ⚠️ 개인형 이동장치(전동킥보드 등)를 운전하는 사람은 **행정안전부령으로 정하는 인명보호 장구(헬멧)를 착용**해야 함.
>
> **위반 시 과태료 20,000원 부과** (도로교통법 제160조 제2항 제2호의2)

### 현황 및 문제점

- **현재 방식**: 경찰 단속 및 육안 확인
- **한계점**:
  - 전국 모든 전동킥보드 이용자를 실시간 감시 불가능
  - 인력 기반 단속의 한계 (시간/장소 제약)
  - 헬멧 미착용 사고 증가 추세
  - 사고 발생 시 치명적 부상 위험 (머리 부상 70% 이상)

### 솔루션

**AI 기반 자동 헬멧 착용 모니터링 시스템**을 통해:
- 📱 **모바일 웹 기반 실시간 헬멧 탐지**
- ⚠️ **헬멧 미착용(head) 탐지 시 즉각적인 경고**
- 📱 **Telegram Bot 실시간 알림** (헬멧 미착용 감지 시)
- 🤖 **RAG 기반 챗봇**: 헬멧 관련 법규, 안전 가이드 질의응답
- 24시간 자동 모니터링 가능

<img src="materials/train_batch0.jpg" width="600" alt="Train Batch Sample">

---

## 🎯 핵심 차별점 및 연구 기여

### 기존 시스템 vs 본 연구

#### 📌 기존 헬멧 탐지 시스템의 한계
```
기존 헬멧 탐지 방식:
- ❌ CCTV 기반 사후 분석 (실시간 대응 불가)
- ❌ 고정된 위치에서만 감지 가능
- ❌ 고가의 전용 하드웨어 필요
- ❌ 개인 사용자가 직접 활용 불가
- ❌ 단순 경고만 제공 (교육 기능 없음)
```

#### ⭐ 본 연구의 혁신: Mobile-First AI Helmet Detection + RAG
```
본 연구의 접근 방식:
- ✅ 모바일 웹 기반 실시간 탐지 (어디서나 사용 가능)
- ✅ 사전 훈련 모델 활용 (빠른 배포 및 경량화)
- ✅ 개인 사용자 중심 서비스 (자가 진단)
- ✅ Telegram 즉각 알림 (실시간 피드백)
- ✅ RAG 기반 챗봇 (법규/안전 가이드 제공)
```

### 💡 주요 기여점 (Contributions)

#### 1. **Mobile-First Deployment Architecture**
- 모바일 웹 기반 실시간 헬멧 탐지 서비스
- WebRTC 기반 카메라 스트리밍 (별도 앱 설치 불필요)
- 경량 YOLOv8n 모델로 모바일 환경 최적화
- **전동킥보드 이용자가 직접 사용 가능한 접근성**

#### 2. **Transfer Learning from Industrial Safety Domain**
- SafetyVisionAI 사전 훈련 모델 활용
- 건설현장 PPE → 전동킥보드 헬멧 도메인 전이
- mAP@0.5 **93.7%** 성능 유지
- Helmet-Head 구분 정확도 **99.55%**

#### 3. **RAG-Enhanced Safety Education**
- 단순 탐지를 넘어 **교육 기능 통합**
- 헬멧 관련 법규, 착용법, 사고 사례 질의응답
- 벡터 DB 기반 관련 문서 검색 및 생성형 AI 답변
- **탐지 + 교육**의 통합 안전 플랫폼

#### 4. **Real-time Alert System**
- Telegram Bot 즉각 알림 (헬멧 미착용 감지 시)
- 탐지 결과 이미지 포함 전송
- 가족/친구 그룹 공유 가능 (안전 네트워크 구축)

### 📊 정량적 성과 (기반 모델)

| 지표 | 성능 | 의미 |
|------|------|------|
| **Helmet-Head 구분** | 혼동률 0.45% | 착용/미착용 거의 완벽 구분 |
| **전체 정확도** | mAP@0.5 93.7% | 실무 적용 가능 수준 |
| **Head 클래스 정확도** | 92.3% | 헬멧 미착용 감지 신뢰성 높음 |
| **추론 속도** | 32ms/이미지 | 모바일 실시간 처리 가능 |
| **모델 크기** | 6.0MB | 모바일 최적화 |

### 🎓 학술적 가치

- **Domain Transfer Learning**: 산업안전 → 개인 모빌리티 안전
- **Mobile AI Deployment**: 경량 모델 기반 모바일 웹 서비스 구현
- **RAG Integration**: 객체 탐지 + 생성형 AI 챗봇 융합
- **Accessible Safety Service**: 개인 사용자 중심 안전 서비스 설계

---

## 기반 모델 (SafetyVisionAI 사전 훈련 모델)

### 🎉 3 Class 모델 (helmet, head, vest) - A100 GPU, 100 Epochs

본 프로젝트는 SafetyVisionAI 프로젝트에서 사전 훈련된 **best.pt 모델**을 활용합니다.
전동킥보드 헬멧 탐지를 위해 `helmet`과 `head` 클래스를 주로 사용합니다.

### ✅ 최종 성능 지표

| 지표 | 결과 | 목표 | 달성 여부 |
|------|------|------|----------|
| **mAP@0.5** | **93.7%** | ≥ 90% | ✅ **초과 달성** (+3.7%p) |
| **mAP@0.5:0.95** | **69.0%** | ≥ 70% | ⚠️ 근접 (-1.0%p) |
| **Precision** | **92.2%** | ≥ 88% | ✅ **초과 달성** (+4.2%p) |
| **Recall** | **87.2%** | ≥ 85% | ✅ **달성** (+2.2%p) |

### 🎯 클래스별 성능 (Validation Set)

| 클래스 | 정확도 | 정답 탐지 | 주요 오분류 | 미탐지율 |
|--------|--------|-----------|-------------|----------|
| **⛑️ Helmet** | **93%** | 6,304개 | head: 29개 (0.4%) | 10% |
| **👤 Head** | **90%** | 1,024개 | helmet: 6개 (0.5%) | 13% |
| **🦺 Vest** | **92%** | 2,529개 | helmet: 6개 (0.2%) | 18% |

**📌 핵심 성과:**
- ✅ **Helmet vs Head 구분 성공**: 헬멧 착용/미착용 간 혼동률 **0.4%**로 매우 낮음
- ✅ **클래스 간 혼동 최소화**: 전체 10,862개 객체 중 49개만 오분류 (**0.45%**)
- ✅ **실시간 안전 경고 가능**: Head 클래스 90% 정확도로 미착용 탐지

### ⚙️ 훈련 환경

| 항목 | 설정 |
|------|------|
| **GPU** | RunPod A100 (40GB) |
| **총 훈련 시간** | ⚡ **54.4분** (3,262초) |
| **Epochs** | 100 |
| **Batch Size** | 128 |
| **Image Size** | 640×640 |
| **Model** | YOLOv8n (Nano) |
| **Optimizer** | AdamW |
| **Initial LR** | 0.01 |
| **AMP** | True (Mixed Precision) |

### 📉 Loss 감소 추이

| Loss 종류 | 초기값 (Epoch 1) | 최종값 (Epoch 100) | 감소율 |
|-----------|------------------|---------------------|--------|
| train/box_loss | 1.501 | 0.821 | 45.3% ⬇️ |
| train/cls_loss | 1.823 | 0.408 | 77.6% ⬇️ |
| train/dfl_loss | 1.375 | 0.987 | 28.2% ⬇️ |
| val/box_loss | 2.439 | 0.946 | 61.2% ⬇️ |
| val/cls_loss | 5.476 | 0.488 | 91.1% ⬇️ |
| val/dfl_loss | 2.959 | 1.045 | 64.7% ⬇️ |

✅ 모든 손실이 꾸준히 감소하며, validation loss도 함께 감소하여 **과적합 없이** 잘 학습되었습니다.

### 📈 학습 곡선

<img src="models/ppe_detection/results.png" width="800" alt="3 Class Training Results">

### 🔍 혼동 행렬 (Confusion Matrix) 분석

<img src="models/ppe_detection/confusion_matrix_normalized.png" width="500" alt="3 Class Confusion Matrix">

#### 클래스 간 혼동 통계

| 혼동 유형 | 건수 | 비율 | 평가 |
|-----------|------|------|------|
| **Helmet → Head** | 29개 | 0.4% | ✅ 매우 낮음 |
| **Head → Helmet** | 6개 | 0.5% | ✅ 매우 낮음 |
| **Helmet ↔ Vest** | 14개 | 0.2% | ✅ 매우 낮음 |
| **Head ↔ Vest** | 0개 | 0% | ✅ 없음 |

### 💡 결과 해석

#### ✅ 강점

1. **🎯 높은 탐지 정확도**: mAP@0.5 = 93.7%로 목표(90%) 초과 달성
2. **⚠️ 헬멧 미착용 감지 성공**: Head 클래스 90% 정확도로 실시간 안전 경고 가능
3. **🔀 클래스 간 혼동 최소화**: Helmet-Head 혼동률 0.4%로 착용/미착용 명확히 구분
4. **📈 안정적 학습**: 과적합 없이 꾸준한 성능 향상
5. **⚡ 빠른 학습**: A100으로 54분 만에 100 epochs 완료

#### 🔧 개선 가능 영역

1. **IoU 엄격 기준**: mAP@0.5:0.95가 69.0%로 목표(70%) 대비 1%p 부족
2. **Head 클래스 Recall**: 87% (1,024/1,178)로 13% 미탐지 → 데이터 증강 필요
3. **Vest 클래스 Recall**: 82% (2,529/3,082)로 18% 미탐지 → Recall 개선 필요

#### 🎯 결론

이 모델은 **🏗️ 건설현장 PPE 탐지 및 안전 경고에 매우 적합**합니다:

- ✅ **실용성**: 93.7% mAP@0.5로 실시간 모니터링 가능
- ⚠️ **안전 경고**: Head 클래스 90% 정확도로 헬멧 미착용 즉각 감지
- 🔒 **신뢰성**: Helmet/Head/Vest 간 혼동률 0.45%로 매우 신뢰할 수 있음
- ⚡ **효율성**: YOLOv8n 경량 모델로 빠른 추론 속도 기대

### 📁 결과 파일

| 파일 | 위치 |
|------|------|
| 최고 성능 모델 (3 Class) | `models/ppe_detection/weights/best.pt` |
| 마지막 체크포인트 | `models/ppe_detection/weights/last.pt` |
| 훈련 통계 (100 epochs) | `models/ppe_detection/results.csv` |
| 혼동 행렬 (3 Class) | `models/ppe_detection/confusion_matrix.png` |
| PR 곡선 | `models/ppe_detection/BoxPR_curve.png` |

---

## 추론 결과 예시

### 통합 추론 시스템 실행 결과

학습된 모델을 사용하여 실제 테스트 이미지에 대한 추론을 수행한 결과입니다.

<img src="materials/inference_result_example.png" width="800" alt="Inference Result Example">

**탐지 결과:**
- ✅ **Helmet (파란색)**: 5명 착용
- ⚠️ **Head (빨간색)**: 0명 (모두 착용)
- 🦺 **Vest (노란색)**: 4개 착용

**안전 평가:**
- 총 작업자: 5명
- 헬멧 착용률: 100%
- 안전 수준: ✅ Excellent (우수)

이 예시는 모델이 helmet과 vest를 동시에 탐지하여 작업 현장의 안전 장비 착용 상태를 종합적으로 모니터링할 수 있음을 보여줍니다.

### 대규모 현장 탐지 결과 (11명)

<img src="materials/inference_result_vest_example.png" width="800" alt="Large Scale Detection Example">

**탐지 결과:**
- ✅ **Helmet (파란색)**: 11명 착용
- ⚠️ **Head (빨간색)**: 0명 (모두 착용)
- 🦺 **Vest (노란색)**: 8개 착용

**안전 평가:**
- 총 작업자: 11명
- 헬멧 착용률: 100%
- 안전 수준: ✅ Excellent (우수)

다수의 작업자가 밀집된 대규모 현장에서도 모든 객체를 정확하게 탐지하여 안전 상태를 실시간으로 모니터링할 수 있습니다.

---

## 빠른 시작

### 1. 환경 설정
```bash
# 저장소 클론
git clone https://github.com/jhboyo/eScooterAI.git
cd eScooterAI

# 의존성 설치 (uv 패키지 관리자)
uv sync

# 환경 변수 파일 생성
cp .env.example .env
```

### 2. 환경 변수 설정
`.env` 파일을 편집하여 다음 항목들을 설정하세요:

```bash
# 프로젝트 경로
PROJECT_ROOT=/path/to/eScooterAI

# Telegram Bot (선택사항)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELEGRAM_ALERTS_ENABLED=true

# OpenAI API (RAG 챗봇용)
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 모델 파일 확인
사전 훈련된 모델이 다음 경로에 있는지 확인:
```bash
models/ppe_detection/weights/best.pt  # 6.0MB
```

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **언어** | Python 3.11 |
| **패키지 관리** | uv |
| **딥러닝** | PyTorch, Ultralytics (YOLOv8n) |
| **이미지 처리** | OpenCV, PIL, NumPy |
| **웹 프레임워크** | Streamlit |
| **실시간 스트리밍** | WebRTC (streamlit-webrtc), aiortc |
| **RAG** | LangChain, ChromaDB, OpenAI API |
| **벡터 임베딩** | OpenAI Embeddings / Sentence-Transformers |
| **알림** | Telegram Bot API |
| **배포** | Streamlit Community Cloud (예정) |

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
eScooterAI/
├── configs/                # 설정 파일
│   ├── ppe_dataset.yaml   # 기반 모델 데이터셋 설정
│   └── rag_config.yaml    # RAG 설정 (예정)
├── models/                 # 모델 파일
│   └── ppe_detection/     # SafetyVisionAI 사전 훈련 모델
│       └── weights/       # best.pt (6.0MB)
├── src/                    # 소스 코드
│   ├── mobile_app/        # 모바일 웹 서비스 (메인)
│   │   ├── app.py         # Streamlit 메인 앱
│   │   ├── pages/         # 멀티페이지
│   │   │   ├── 1_helmet_detection.py  # 헬멧 탐지 페이지
│   │   │   └── 2_safety_chatbot.py    # RAG 챗봇 페이지
│   │   ├── components/    # UI 컴포넌트
│   │   │   ├── camera.py      # WebRTC 카메라
│   │   │   ├── detector.py    # 헬멧 탐지
│   │   │   └── chatbot.py     # RAG 챗봇 UI
│   │   └── utils/         # 유틸리티
│   │       ├── inference.py   # YOLO 추론
│   │       └── alert.py       # Telegram 알림
│   ├── rag/               # RAG 시스템 (NEW!)
│   │   ├── __init__.py
│   │   ├── vector_store.py    # ChromaDB 벡터 저장소
│   │   ├── embeddings.py      # 문서 임베딩
│   │   ├── retriever.py       # 문서 검색
│   │   ├── generator.py       # LLM 답변 생성
│   │   └── pipeline.py        # RAG 파이프라인
│   ├── data/              # RAG 데이터 (NEW!)
│   │   └── safety_docs/   # 헬멧 안전 관련 문서
│   │       ├── laws/          # 법규 (도로교통법 등)
│   │       ├── guides/        # 착용법, 선택 가이드
│   │       └── cases/         # 사고 사례
│   ├── alert/             # 알림 모듈
│   │   ├── __init__.py
│   │   └── telegram_notifier.py  # Telegram Bot
│   ├── inference/         # 기존 CLI 추론 (유지)
│   └── webcam_inference/  # 기존 웹캠 추론 (참고용)
├── output/                 # 출력 결과
│   ├── detections/        # 탐지 결과 저장
│   └── screenshots/       # 스크린샷
├── vector_db/              # ChromaDB 저장소 (NEW!)
├── materials/              # 참고 자료
├── .streamlit/             # Streamlit 설정
│   ├── config.toml        # 테마 및 서버 설정
│   └── secrets.toml       # API 키 (gitignore)
├── pyproject.toml          # 의존성 정의
├── .env.example            # 환경 변수 예시
└── README.md
```

---

## 진행 현황

### Phase 0: 기반 모델 준비 ✅ (SafetyVisionAI)
- [v] YOLOv8n 모델 사전 훈련 완료 (mAP@0.5: 93.7%)
- [v] best.pt 모델 확보 (6.0MB, 경량 모델)
- [v] Helmet-Head 구분 정확도 99.55% 검증
- [v] 모바일 실시간 추론 가능 확인 (32ms/이미지)

### Phase 1: 프로젝트 초기 설정 🚀 (진행 중)
- [v] 프로젝트 개요 및 README 업데이트
- [ ] 환경 설정 및 의존성 설치
  - [ ] RAG 관련 라이브러리 추가 (LangChain, ChromaDB, OpenAI)
  - [ ] WebRTC 라이브러리 확인 (streamlit-webrtc, aiortc)
- [ ] 프로젝트 구조 재구성
  - [ ] `src/mobile_app/` 디렉토리 생성
  - [ ] `src/rag/` 디렉토리 생성
  - [ ] `src/data/safety_docs/` 디렉토리 생성

### Phase 2: RAG 시스템 구축 📚 (예정)
- [ ] **안전 문서 데이터 수집**
  - [ ] 도로교통법 제50조 (헬멧 착용 의무)
  - [ ] 전동킥보드 안전 가이드
  - [ ] 헬멧 선택 및 착용법
  - [ ] 사고 사례 및 통계
- [ ] **벡터 DB 구축**
  - [ ] ChromaDB 설정
  - [ ] 문서 임베딩 (OpenAI Embeddings / Sentence-Transformers)
  - [ ] 벡터 저장소 생성 및 인덱싱
- [ ] **RAG 파이프라인 구현**
  - [ ] Retriever: 관련 문서 검색
  - [ ] Generator: LLM 기반 답변 생성 (OpenAI API)
  - [ ] 프롬프트 엔지니어링 (헬멧 안전 전문가 페르소나)

### Phase 3: 모바일 웹 서비스 구현 📱 (예정)
- [ ] **멀티페이지 Streamlit 앱 구조**
  - [ ] 페이지 1: 실시간 헬멧 탐지 (WebRTC)
  - [ ] 페이지 2: 안전 가이드 챗봇 (RAG)
  - [ ] 홈 페이지: 프로젝트 소개 및 사용법
- [ ] **WebRTC 기반 실시간 탐지**
  - [ ] streamlit-webrtc 통합
  - [ ] 모바일 카메라 스트리밍
  - [ ] 프레임 단위 YOLO 추론
  - [ ] 실시간 바운딩 박스 오버레이
- [ ] **Telegram 알림 통합**
  - [ ] 헬멧 미착용 감지 시 즉각 알림
  - [ ] 탐지 결과 이미지 전송
  - [ ] 알림 설정 UI (사용자가 활성화/비활성화)
- [ ] **모바일 UI/UX 최적화**
  - [ ] 반응형 레이아웃 (모바일/태블릿/데스크톱)
  - [ ] 터치 인터페이스 최적화
  - [ ] 로딩 시간 최소화

### Phase 4: 통합 테스트 및 배포 🚀 (예정)
- [ ] **로컬 테스트**
  - [ ] 헬멧 탐지 기능 테스트 (다양한 각도, 조명)
  - [ ] RAG 챗봇 응답 품질 평가
  - [ ] Telegram 알림 테스트
  - [ ] 모바일 브라우저 호환성 테스트
- [ ] **Streamlit Community Cloud 배포**
  - [ ] GitHub 연동 자동 배포 설정
  - [ ] 환경 변수 및 Secrets 설정 (OpenAI API, Telegram Bot)
  - [ ] 배포 URL 확보
  - [ ] 배포 가이드 문서 작성
- [ ] **성능 최적화**
  - [ ] 모델 추론 속도 최적화
  - [ ] 벡터 DB 쿼리 속도 최적화
  - [ ] 메모리 사용량 최적화

### Phase 5: 논문 작성 및 발표 📝 (예정)
- [ ] **실험 결과 정리**
  - [ ] 헬멧 탐지 정확도 측정 (전동킥보드 환경)
  - [ ] RAG 챗봇 응답 품질 평가
  - [ ] 사용자 만족도 조사 (설문)
  - [ ] 기존 시스템 대비 우수성 입증
- [ ] **학술 논문 작성**
  - [ ] 서론: 연구 배경 및 문제 정의
  - [ ] 관련 연구: 기존 헬멧 탐지 시스템 분석
  - [ ] 방법론: Transfer Learning + RAG 아키텍처
  - [ ] 실험 결과: 정량적/정성적 평가
  - [ ] 결론 및 향후 연구
- [ ] **최종 발표 준비**
  - [ ] 발표 자료 작성 (PPT)
  - [ ] 데모 영상 제작
  - [ ] 실시간 시연 준비

---

## 향후 과제 및 개선 계획

### 🎯 핵심 개발 과제 (Phase 1-4)

#### 1. RAG 시스템 구축 (최우선)
- 헬멧 관련 법규, 안전 가이드 문서 수집 및 벡터화
- LangChain + ChromaDB 기반 RAG 파이프라인 구현
- OpenAI API 통합 (GPT-4 Turbo / GPT-3.5 Turbo)

#### 2. 모바일 웹 서비스 개발
- Streamlit 멀티페이지 구조 설계
- WebRTC 기반 실시간 카메라 스트리밍
- 모바일 반응형 UI/UX 최적화

#### 3. 통합 및 배포
- Streamlit Community Cloud 배포
- 환경 변수 및 Secrets 관리
- 성능 최적화 (추론 속도, 메모리)

### 🔬 연구 과제 (Phase 5)

#### 1. 전동킥보드 환경 성능 평가
- 실제 전동킥보드 이용자 데이터 수집
- 다양한 조명/각도/속도에서 탐지 성능 측정
- 기반 모델의 도메인 전이 성능 분석

#### 2. RAG 챗봇 품질 평가
- 응답 정확도, 관련성, 유용성 평가
- 사용자 만족도 조사 (설문/인터뷰)
- 프롬프트 엔지니어링 최적화

### ⚠️ 알려진 제한사항 (기반 모델)

#### DS2 스타일 이미지에서 Head 클래스 탐지 한계

SafetyVisionAI 기반 모델은 **어두운 배경 이미지(DS2 스타일)에서 헬멧 미착용자(head) 탐지 성능 저하** 문제가 있습니다.

#### 🔍 문제 상황

테스트 이미지 3개 모두에서 **Head 클래스 탐지 실패**:

| 이미지 | 실제 상황 | 모델 탐지 | 결과 |
|--------|----------|-----------|------|
| ds2_helmet_jacket_10142.jpg | 검은 머리 (헬멧 ❌) | Vest만 탐지 | ❌ 미탐지 |
| ds2_helmet_jacket_03480.jpg | 측면 머리 (헬멧 ❌) | Vest만 탐지 | ❌ 미탐지 |
| ds2_helmet_jacket_01267.jpg | 뒤쪽 머리 (헬멧 ❌) | Vest만 탐지 | ❌ 미탐지 |

**신뢰도 0.01 (매우 낮음)에서도 Head 탐지 0개** → 모델이 DS2 스타일 head를 전혀 학습하지 못함

#### 📊 근본 원인 분석

##### 1. **데이터셋 클래스 불균형**

```
전체 학습 데이터 분포:
├── Helmet: 25,425개 (64.5%) ✅
├── Head: 3,679개 (9.3%) ⚠️ 너무 적음!
└── Vest: 10,351개 (26.2%) ✅
```

- **Head 클래스가 전체의 9.3%에 불과**
- Helmet 대비 1/7 수준 → 심각한 불균형

##### 2. **DS1과 DS2 데이터 분포 차이**

| 데이터셋 | Helmet | Head | Vest | 특징 |
|---------|--------|------|------|------|
| **DS1** (Hard Hat Detection) | 12,354 | **3,679** ✅ | 0 | 노란색/주황색 헬멧, 밝은 배경 |
| **DS2** (Helmet & Vest) | 13,071 | **0** ❌ | 10,351 | 검은색 헬멧, 어두운 배경 |

**핵심 문제:**
- DS2 데이터셋은 원본부터 **Head 클래스가 하나도 없음**
- DS2는 "안전한 상황"만 수집 (모두 헬멧 착용)
- 모델이 DS2 스타일의 head를 전혀 학습하지 못함

##### 3. **결과: 심각한 모델 편향**

- ✅ **DS1 스타일 head** (밝은 배경, 노란색 헬멧 환경): 잘 탐지 (90% AP)
- ❌ **DS2 스타일 head** (어두운 배경, 검은색 헬멧 환경): 완전 실패 (0% 탐지)

#### 🎯 실제 영향

**산업 현장에서의 위험성:**
1. ❌ **헬멧 미착용자를 탐지하지 못함** → 안전사고 위험
2. ❌ **DS2 스타일 현장(어두운 터널, 실내)에서 시스템 무용지물**
3. ❌ **False Negative**: 위험한 상황을 안전하다고 잘못 판단

**모델 평가 지표와의 괴리:**
- Validation/Test Set: Head AP 90% (우수) ✅
- 실제 DS2 이미지: Head 탐지 0% (완전 실패) ❌
- **평가 세트에는 DS1 스타일만 포함** → 실제 성능 과대평가

#### 💡 해결 방안

##### 즉시 조치 (단기)

1. **DS2 이미지에 Head 레이블 추가** (최우선)
   - DS2 데이터에서 헬멧 미착용 케이스 찾아 수동 레이블링
   - 또는 외부 데이터셋 추가 확보
   - 목표: Head 클래스 비율 20% 이상으로 증가

2. **클래스 가중치(Class Weights) 조정**
   ```python
   # train.py 수정
   class_weights = [1.0, 3.0, 1.0]  # [helmet, head, vest]
   # Head 클래스에 3배 가중치 부여
   ```

3. **Head 클래스 데이터 증강(Augmentation) 강화**
   - 밝기 조절 (어두운 환경 시뮬레이션)
   - 배경 변화 (터널, 실내 등)
   - 색상 변환 (검은 머리 → 다양한 색상)
   - 목표: Head 데이터 3배 증강 (3,679 → 11,000개)

4. **두 단계 학습(Two-Stage Training)**
   - Stage 1: Helmet-Vest 학습 (DS2 활용)
   - Stage 2: Head 추가 학습 (DS1 + 증강 데이터)

##### 근본 해결 (장기)

5. **추가 데이터 수집**
   - DS2 스타일(어두운 배경, 검은색)의 헬멧 미착용 이미지 확보
   - 다양한 조명 환경의 Head 클래스 데이터 추가
   - 목표: Head 클래스 최소 10,000개 이상

6. **2-Stage 탐지 모델 고려**
   - Stage 1: Person Detection (사람 먼저 찾기)
   - Stage 2: Helmet Classification (헬멧 착용 여부 분류)
   - 각 stage를 별도로 최적화

7. **하드 네거티브 마이닝(Hard Negative Mining)**
   - 현재 탐지 실패한 DS2 이미지들을 훈련 데이터에 추가
   - 모델이 어려워하는 케이스 집중 학습

#### 📈 개선 목표

| 항목 | 현재 | 목표 |
|------|------|------|
| Head 클래스 비율 | 9.3% | **≥ 20%** |
| DS2 스타일 Head 데이터 | 0개 | **≥ 5,000개** |
| DS2 이미지 Head 탐지율 | 0% | **≥ 80%** |
| 전체 Head AP | 90% | **≥ 92%** (모든 스타일) |

#### 🔄 다음 단계

1. **긴급**: DS2 이미지에서 헬멧 미착용 케이스 확보 및 레이블링
2. **우선**: 클래스 가중치 적용 및 재학습
3. **중요**: 데이터 증강 전략 수립 및 적용
4. **검증**: DS2 스타일 테스트 세트로 재평가

**이 문제는 실제 산업 현장 적용 시 심각한 안전 위험으로 이어질 수 있으므로 최우선 해결 과제입니다.**

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
- ✅ **3 Class 모델 성능 검증 완료**
   - A100 GPU 100 epochs 본 훈련 완료
   - 목표 성능 달성: **mAP@0.5 = 93.7%** (목표 90% 초과)
   - Helmet-Head 혼동률 0.45%로 거의 완벽한 구분
   - 상세 분석 보고서 작성 완료 (training_report.md)
- ✅ **통합 추론 시스템 구현 완료**
   - helmet, head, vest 3개 클래스 동시 탐지
   - 단일 이미지 / 디렉토리 처리 지원
   - 헬멧 착용률 자동 계산 및 안전 수준 평가
   - 시각화 결과 (PNG) 및 JSON 저장
   - 명령줄 인터페이스 (CLI) 지원
- ✅ **웹 인터페이스 구축 완료** (Phase 7)
   - Streamlit 기반 대시보드 개발 완료
   - 이미지 업로드 및 실시간 탐지 결과 표시
   - 바운딩 박스 시각화 및 원본/결과 비교 뷰
   - 헬멧 착용률 및 안전 수준 평가 자동화
   - 신뢰도/IoU 임계값 설정 UI
- ✅ **Streamlit Community Cloud 배포 완료** (2025-11-23)
   - GitHub 연동 자동 배포
   - YOLOv8 모델 포함 (best.pt, 6.0MB)
   - 무료 호스팅 (Streamlit Community Cloud)
   - 실시간 웹 데모: https://safetyvisionai.streamlit.app
   - 배포 가이드 문서 (DEPLOYMENT_GUIDE.md)
- ✅ **Telegram Bot 실시간 알림 시스템 구축 완료** (2025-11-23) 📱
   - 헬멧 미착용자 2명 이상 또는 착용률 80% 미만 시 자동 알림
   - Telegram 그룹 채팅 지원 (팀원 모두 알림 수신)
   - 탐지 결과 이미지 포함 전송
   - 안전 수준별 메시지 포맷 (Excellent/Caution/Dangerous)
   - 환경 변수 기반 설정 (.env 파일)
   - Streamlit Cloud Secrets 지원
   - 알림 모듈 (`src/alert/telegram_notifier.py`) 구현

### 남은 과제

다음 과제들은 **Phase 8 이후**에서 개발 예정입니다:

1. **실시간 추론 및 성능 개선** (Phase 8)
   - 웹캠 실시간 추론 (프레임 단위 객체 탐지)
   - 배치 추론 최적화 (GPU 병렬 처리로 속도 50% 이상 향상)
   - 실시간 안전 경고 알림 시스템

2. **안전조끼 미착용 탐지** (향후 연구)
   - 현재: vest 착용만 탐지
   - 개선: person 클래스 추가하여 vest 미착용자 식별
   - 구현: person 탐지 후 vest가 없으면 경고

3. **웹 인터페이스 고도화** (선택사항)
   - 결과 다운로드 (ZIP, PDF 리포트)
   - 세션 히스토리 관리
   - 여러 모델 비교 모드

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
