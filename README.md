# eScooterAI - RAG 기반 전동킥보드 헬멧 안전 통합 플랫폼

**딥러닝 객체 탐지 + RAG 기반 자연어 처리 융합 시스템**

전동킥보드 헬멧 착용 실시간 탐지 및 RAG(Retrieval-Augmented Generation) 기반 안전 교육 챗봇 통합 플랫폼

---

## 🚀 프로젝트 데모

[![Streamlit Cloud](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://escooter-helmet-detection.streamlit.app)

**👉 실시간 웹캠 데모 + RAG 챗봇** (예정)

### 🎯 주요 기능

#### 1. **RAG 기반 안전 가이드 챗봇** 🤖 (NLP 핵심)
헬멧 관련 법규, 착용법, 사고 사례를 자연어로 질문하세요!
- **질문 예시**:
  - "헬멧을 안 쓰면 과태료가 얼마인가요?"
  - "헬멧을 올바르게 착용하는 방법을 알려주세요"
  - "헬멧 착용의 효과는 무엇인가요?"
- **벡터 DB 검색**: ChromaDB 기반 의미적 문서 검색
- **LLM 답변 생성**: GPT-4 기반 컨텍스트 인식 답변
- **환각 방지**: 문서 기반 사실만 전달

#### 2. **실시간 헬멧 탐지** 📱 (CV 모듈)
모바일/웹 브라우저에서 바로 헬멧 탐지를 테스트해보세요!
- 📱 모바일 카메라 실시간 탐지 (WebRTC)
- 🎯 헬멧 착용/미착용 자동 감지 (YOLOv8n)
- 📊 안전 수준 자동 평가
- 🖼️ 실시간 객체 탐지 시각화

#### 3. **Telegram 즉각 알림** 📱 (통합 시스템)
- ⚠️ 헬멧 미착용 감지 시 **즉각 알림 전송**
- 📸 탐지 결과 이미지 포함
- 📊 안전 통계 및 권장 사항

> 💡 **CV + NLP 융합 플랫폼**: 탐지(Detection) → 알림(Alert) → 교육(Education) 통합 파이프라인

---

## 팀 정보

- **팀명**: eScooterAI
- **프로젝트**: 딥러닝 기반 전동킥보드 헬멧 착용 모니터링 모바일 서비스
- **멤버**: 김상진, 김준호, 김한솔, 유승근, 홍준재

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목표** | 전동킥보드 이용자의 헬멧 착용 실시간 감지 + RAG 기반 안전 교육 |
| **탐지 대상** | 헬멧 착용(helmet), 헬멧 미착용(head) |
| **객체 탐지 모델** | YOLOv8n (SafetyVisionAI 사전 훈련 모델 활용) |
| **NLP 시스템** | **RAG (Retrieval-Augmented Generation) 기반 질의응답** |
| **벡터 DB** | ChromaDB (문서 임베딩 및 유사도 검색) |
| **LLM** | OpenAI GPT-4/GPT-3.5 Turbo (답변 생성) |
| **플랫폼** | 모바일 웹 서비스 (Streamlit + WebRTC) |

### 🔬 NLP 연구 핵심
- **Semantic Search**: 벡터 임베딩 기반 의미적 문서 검색
- **Domain-Specific QA**: 헬멧 안전 도메인 특화 질의응답 시스템
- **Context-Aware Generation**: 검색된 문서 컨텍스트 기반 답변 생성
- **Prompt Engineering**: 안전 전문가 페르소나 프롬프트 설계

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

#### ⭐ 본 연구의 혁신: Computer Vision + NLP Fusion Platform
```
본 연구의 접근 방식:
- ✅ CV + NLP 융합: 객체 탐지 + RAG 기반 대화형 교육
- ✅ RAG 시스템: 벡터 DB 기반 의미적 문서 검색 + LLM 생성
- ✅ Domain-Specific QA: 헬멧 안전 특화 질의응답 시스템
- ✅ Mobile-First: 모바일 웹 기반 실시간 탐지 (어디서나 사용)
- ✅ Transfer Learning: SafetyVisionAI 사전 훈련 모델 활용
- ✅ Real-time Alert: Telegram 즉각 알림 (실시간 피드백)
```

### 💡 주요 기여점 (Contributions)

#### 1. **RAG-based Domain-Specific Question Answering System** 🔬
**자연어 처리 핵심 연구**

- **Semantic Document Retrieval**:
  - ChromaDB 벡터 저장소 기반 의미적 문서 검색
  - OpenAI Embeddings / Sentence-Transformers 임베딩
  - Cosine Similarity 기반 Top-K 검색

- **Context-Aware Answer Generation**:
  - 검색된 문서를 컨텍스트로 LLM에 전달
  - GPT-4/GPT-3.5 Turbo 기반 답변 생성
  - Hallucination 방지 (문서 기반 답변 제한)

- **Prompt Engineering**:
  - 헬멧 안전 전문가 페르소나 설계
  - Few-shot Learning 예시 포함
  - Chain-of-Thought 추론 유도

- **Domain-Specific Knowledge Base**:
  - 도로교통법, 헬멧 착용법, 사고 사례 등 구조화
  - 법규, 가이드, 사례 카테고리 분류
  - 문서 청크 최적화 (512 tokens)

**NLP 평가 지표**:
- Retrieval Precision@K (검색 정확도)
- Answer Relevance Score (답변 관련성)
- Semantic Similarity (의미 유사도)
- User Satisfaction Survey (사용자 만족도)

#### 2. **Computer Vision + NLP Multimodal Fusion**
- **CV 모듈**: YOLOv8n 실시간 헬멧 탐지
- **NLP 모듈**: RAG 기반 질의응답 시스템
- **통합 플랫폼**: 탐지 → 알림 → 교육 파이프라인
- **시너지 효과**: 탐지 결과 기반 맞춤형 안전 교육 제공

#### 3. **Transfer Learning from Industrial Safety Domain**
- SafetyVisionAI 사전 훈련 모델 활용
- 건설현장 PPE → 전동킥보드 헬멧 도메인 전이
- mAP@0.5 **93.7%** 성능 유지
- Helmet-Head 구분 정확도 **99.55%**

#### 4. **Mobile-First Deployment Architecture**
- 모바일 웹 기반 실시간 헬멧 탐지 서비스
- WebRTC 기반 카메라 스트리밍 (별도 앱 설치 불필요)
- 경량 YOLOv8n 모델로 모바일 환경 최적화
- **전동킥보드 이용자가 직접 사용 가능한 접근성**

#### 5. **Real-time Alert System**
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

#### 자연어 처리 (NLP) 연구
- **RAG Architecture**: Retrieval-Augmented Generation 기반 질의응답 시스템
- **Semantic Search**: 벡터 임베딩 기반 의미적 문서 검색 최적화
- **Domain Adaptation**: 헬멧 안전 도메인 특화 지식베이스 구축
- **Prompt Engineering**: 안전 전문가 페르소나 기반 프롬프트 설계
- **Hallucination Mitigation**: 문서 기반 답변 제한을 통한 환각 방지
- **Evaluation Metrics**: Retrieval Precision, Answer Relevance, Semantic Similarity

#### 컴퓨터 비전 (CV) + 멀티모달 융합
- **Domain Transfer Learning**: 산업안전 (PPE) → 개인 모빌리티 안전 (헬멧)
- **Multimodal Integration**: CV (탐지) + NLP (교육) 융합 플랫폼
- **Mobile AI Deployment**: 경량 모델 기반 모바일 웹 서비스 구현

#### 실용성 및 접근성
- **End-to-End Pipeline**: 탐지 → 알림 → 교육 통합 시스템
- **Accessible Safety Service**: 개인 사용자 중심 안전 서비스 설계
- **Reproducible Research**: 공개 모델 및 RAG 파이프라인 기반 재현 가능

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

#### 2.1 안전 문서 데이터 수집 및 전처리
- [ ] **법규 문서**
  - [ ] 도로교통법 제50조 (헬멧 착용 의무)
  - [ ] 도로교통법 제160조 (과태료 규정)
  - [ ] 개인형 이동장치 안전기준 고시
- [ ] **안전 가이드**
  - [ ] 헬멧 선택 기준 (인증 마크, 크기, 재질)
  - [ ] 헬멧 올바른 착용법 (각도, 턱끈, 조절)
  - [ ] 전동킥보드 안전 운전 수칙
- [ ] **사고 사례 및 통계**
  - [ ] 헬멧 미착용 사고 통계 (교통안전공단)
  - [ ] 헬멧 착용 효과 연구 결과
  - [ ] 실제 사고 사례 분석
- [ ] **문서 전처리**
  - [ ] PDF/웹 크롤링 및 텍스트 추출
  - [ ] 문서 청크 분할 (512 tokens, overlap 50)
  - [ ] 메타데이터 태깅 (카테고리, 출처, 날짜)

#### 2.2 벡터 DB 구축 (ChromaDB)
- [ ] **임베딩 모델 선택 및 비교**
  - [ ] OpenAI text-embedding-3-small (성능 우선)
  - [ ] Sentence-Transformers paraphrase-multilingual (무료 대안)
  - [ ] 한국어 도메인 성능 벤치마크
- [ ] **ChromaDB 설정**
  - [ ] Collection 생성 (helmet_safety_docs)
  - [ ] 문서 임베딩 및 저장
  - [ ] 인덱싱 최적화 (HNSW 알고리즘)
- [ ] **검색 성능 최적화**
  - [ ] Top-K 파라미터 튜닝 (K=3~5)
  - [ ] Similarity Threshold 설정 (>0.7)
  - [ ] Re-ranking 알고리즘 적용 (선택사항)

#### 2.3 RAG 파이프라인 구현 (LangChain)
- [ ] **Retriever 구현**
  - [ ] Query Embedding 생성
  - [ ] Cosine Similarity 기반 Top-K 검색
  - [ ] 검색 결과 필터링 및 정렬
- [ ] **Generator 구현**
  - [ ] OpenAI API 통합 (GPT-4 Turbo / GPT-3.5 Turbo)
  - [ ] Context 구성 (검색 문서 + 사용자 질문)
  - [ ] Temperature, Max Tokens 설정
- [ ] **Prompt Engineering**
  - [ ] System Prompt: 헬멧 안전 전문가 페르소나
  - [ ] Few-shot Examples: 질문-답변 예시 3~5개
  - [ ] Chain-of-Thought: 단계별 추론 유도
  - [ ] Output Format: 답변 구조화 (근거 + 핵심 답변 + 추가 정보)
- [ ] **RAG Chain 통합**
  - [ ] LangChain LCEL 기반 파이프라인 구축
  - [ ] Retrieval → Context → Generation 자동화
  - [ ] 에러 핸들링 및 폴백 메커니즘

#### 2.4 RAG 평가 및 최적화
- [ ] **검색 성능 평가**
  - [ ] Retrieval Precision@K (K=3, 5, 10)
  - [ ] Recall@K (관련 문서 검색율)
  - [ ] MRR (Mean Reciprocal Rank)
- [ ] **답변 품질 평가**
  - [ ] Answer Relevance Score (LLM as Judge)
  - [ ] Semantic Similarity (답변-정답 유사도)
  - [ ] Factual Consistency (문서 기반 사실 일치도)
  - [ ] Hallucination Rate (환각 발생률)
- [ ] **사용자 평가**
  - [ ] User Satisfaction Survey (5점 척도)
  - [ ] Response Time (응답 속도)
  - [ ] Usefulness Rating (답변 유용성)

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

#### 5.1 실험 결과 정리

**컴퓨터 비전 실험**
- [ ] 헬멧 탐지 정확도 측정 (전동킥보드 환경)
  - [ ] 실제 전동킥보드 이용자 테스트 영상 수집
  - [ ] 다양한 조명/각도/속도에서 탐지 성능
  - [ ] mAP@0.5, Precision, Recall 측정
  - [ ] Transfer Learning 효과 분석

**자연어 처리 실험 (핵심)**
- [ ] **RAG 검색 성능 평가**
  - [ ] Test Set 구성 (질문-정답 쌍 50~100개)
  - [ ] Retrieval Precision@K (K=3, 5, 10) 측정
  - [ ] Recall@K 및 MRR 계산
  - [ ] 임베딩 모델 비교 (OpenAI vs Sentence-Transformers)
- [ ] **RAG 답변 품질 평가**
  - [ ] Answer Relevance Score (GPT-4 as Judge, 5점 척도)
  - [ ] Semantic Similarity (답변-정답 코사인 유사도)
  - [ ] Factual Consistency (문서 기반 사실 일치도)
  - [ ] Hallucination Rate (환각 발생률, < 10% 목표)
  - [ ] Response Time (평균 응답 속도, < 3초 목표)
- [ ] **프롬프트 엔지니어링 효과**
  - [ ] Baseline (프롬프트 없음) vs 전문가 페르소나
  - [ ] Few-shot Learning 예시 개수별 비교 (0, 3, 5개)
  - [ ] Chain-of-Thought 유무 비교
- [ ] **사용자 평가 (User Study)**
  - [ ] 사용자 만족도 설문 (5점 척도, N=20~30명)
  - [ ] 답변 유용성, 이해도, 신뢰도 평가
  - [ ] 기존 검색 엔진 대비 선호도

**통합 시스템 평가**
- [ ] End-to-End 성능 측정 (탐지 → 알림 → 챗봇)
- [ ] 기존 시스템 대비 우수성 입증
- [ ] 사용 시나리오별 효과성 분석

#### 5.2 학술 논문 작성

**논문 구조 (한국어/영어)**
- [ ] **Abstract (초록)**
  - [ ] 연구 배경 및 동기
  - [ ] RAG 기반 접근 방식 요약
  - [ ] 주요 결과 (정량적 지표 포함)
- [ ] **1. Introduction (서론)**
  - [ ] 전동킥보드 헬멧 미착용 문제
  - [ ] 기존 탐지 시스템의 한계
  - [ ] RAG 기반 교육 통합의 필요성
  - [ ] 연구 목표 및 기여점
- [ ] **2. Related Work (관련 연구)**
  - [ ] 헬멧 탐지 시스템 (YOLO 기반)
  - [ ] RAG 시스템 (LangChain, ChromaDB)
  - [ ] Domain-Specific QA Systems
  - [ ] Transfer Learning in Safety Domain
- [ ] **3. Methodology (방법론)**
  - [ ] 3.1 System Architecture (전체 아키텍처)
  - [ ] 3.2 Helmet Detection Module (YOLOv8n)
  - [ ] 3.3 **RAG-based QA Module (핵심)**
    - [ ] Document Collection & Preprocessing
    - [ ] Vector Embedding (ChromaDB, OpenAI)
    - [ ] Semantic Retrieval (Cosine Similarity)
    - [ ] Context-Aware Generation (GPT-4)
    - [ ] Prompt Engineering Strategy
  - [ ] 3.4 Integration & Deployment (통합)
- [ ] **4. Experiments (실험)**
  - [ ] 4.1 Experimental Setup (실험 환경)
  - [ ] 4.2 Helmet Detection Results (CV 결과)
  - [ ] 4.3 **RAG System Evaluation (NLP 결과 - 핵심)**
    - [ ] Retrieval Performance
    - [ ] Answer Quality Metrics
    - [ ] Prompt Engineering Effects
    - [ ] User Study Results
  - [ ] 4.4 Ablation Study (제거 실험)
  - [ ] 4.5 Comparison with Baselines (베이스라인 비교)
- [ ] **5. Discussion (논의)**
  - [ ] 연구 결과 해석
  - [ ] RAG 시스템의 강점 및 한계
  - [ ] 실용적 함의 (Practical Implications)
  - [ ] 한계점 및 개선 방향
- [ ] **6. Conclusion (결론)**
  - [ ] 연구 요약
  - [ ] 주요 기여점 재강조
  - [ ] 향후 연구 방향
- [ ] **References (참고문헌)**
  - [ ] YOLO, RAG, LangChain 관련 논문
  - [ ] 헬멧 안전, 전동킥보드 관련 연구

#### 5.3 최종 발표 준비
- [ ] **발표 자료 작성 (PPT)**
  - [ ] 연구 배경 및 동기 (3분)
  - [ ] RAG 시스템 아키텍처 (5분)
  - [ ] 실험 결과 (정량적 지표 중심, 7분)
  - [ ] 데모 시연 (3분)
  - [ ] 결론 및 질의응답 (2분)
- [ ] **데모 영상 제작**
  - [ ] 헬멧 탐지 실시간 데모
  - [ ] RAG 챗봇 질의응답 예시
  - [ ] 통합 시스템 사용 시나리오
- [ ] **실시간 시연 준비**
  - [ ] 모바일 웹 앱 배포 URL
  - [ ] 백업 데모 영상 준비
  - [ ] 예상 질문 답변 준비

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

## 개발 일정

| 단계 | 기간 | 주요 목표 | 핵심 Deliverable | 상태 |
|------|------|----------|------------------|------|
| **Phase 1** | Week 1 (11/28~) | 프로젝트 초기화, 구조 재구성 | README, 디렉토리 구조 | 🚀 진행 중 |
| **Phase 2** | Week 2-3 | **RAG 시스템 구축 (NLP 핵심)** | **벡터 DB, QA 파이프라인** | 📅 예정 |
|  | Week 2 | 문서 수집 및 전처리 | 안전 문서 데이터셋 (법규, 가이드, 사례) |  |
|  | Week 2-3 | ChromaDB 벡터화 | 임베딩 모델, 검색 성능 벤치마크 |  |
|  | Week 3 | LangChain RAG 파이프라인 | Retriever + Generator + Prompt |  |
|  | Week 3 | **RAG 평가 실험** | **Precision@K, Relevance, Hallucination** |  |
| **Phase 3** | Week 4 | 모바일 웹 서비스 개발 | Streamlit 멀티페이지 앱 | 📅 예정 |
|  |  | WebRTC 통합 | 실시간 카메라 스트리밍 |  |
|  |  | RAG 챗봇 UI | 질의응답 인터페이스 |  |
| **Phase 4** | Week 5 | 통합 테스트 및 배포 | Streamlit Cloud 배포 URL | 📅 예정 |
|  |  | End-to-End 테스트 | 탐지 → 알림 → 챗봇 통합 |  |
| **Phase 5** | Week 6-7 | **실험, 논문 작성, 발표** | **학술 논문, 발표 자료** | 📅 예정 |
|  | Week 6 | **RAG 성능 평가 실험** | **NLP 정량적 지표 측정** |  |
|  | Week 6-7 | 논문 작성 | 서론, 방법론, 실험, 결론 |  |
|  | Week 7 | 발표 준비 | PPT, 데모 영상, 시연 |  |

---

## 참고 자료

### 법규 및 규정
- [도로교통법 제50조](https://www.law.go.kr/) (개인형 이동장치 안전기준)
- 도로교통법 제160조 (헬멧 미착용 과태료)
- 개인형 이동장치 안전기준 고시 (국토교통부)

### RAG 및 자연어 처리 (NLP) 🔬
- **RAG 논문**: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- **LangChain**: [LangChain Documentation](https://python.langchain.com/) - RAG 파이프라인 구축
- **ChromaDB**: [ChromaDB Documentation](https://docs.trychroma.com/) - 벡터 임베딩 및 검색
- **OpenAI Embeddings**: [text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings) - 문서 벡터화
- **Sentence-Transformers**: [paraphrase-multilingual-MiniLM](https://huggingface.co/sentence-transformers) - 다국어 임베딩
- **Prompt Engineering**: [OpenAI Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)
- **RAG Evaluation**: [RAGAS Framework](https://docs.ragas.io/) - RAG 평가 지표 (Precision, Relevance, Hallucination)

### 컴퓨터 비전 (CV)
- **YOLO**: [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- **Transfer Learning**: [Fine-tuning Pre-trained Models](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- 논문: `YOLO(You Only Look Once) 모델별 건설 현장 위험 상태 및 객체 인식 성능 비교`

### 모바일 웹 및 배포
- **Streamlit**: [Streamlit Documentation](https://docs.streamlit.io/)
- **WebRTC**: [streamlit-webrtc GitHub](https://github.com/whitphx/streamlit-webrtc) - 실시간 카메라 스트리밍
- **aiortc**: [aiortc Documentation](https://aiortc.readthedocs.io/) - Python WebRTC 구현

### 관련 연구 (헬멧 탐지 및 안전)
- 논문: `딥 러닝 기반 작업자 개인보호구 착용 및 얼굴 신원 확인 시스템에 관한 연구`
- 특허: `인공지능기반 이륜자동차의 헬맷 미착용 단속시스템 및 방법` (KR 특허)
- 논문: `Construction Site Hazards Identification Using Deep Learning and Computer Vision`
- 논문: `SYSTEM AND METHOD FOR AI VISUAL INSPECTION`

### 기반 프로젝트
- [SafetyVisionAI](https://github.com/jhboyo/SafetyVisionAI) - YOLOv8n 사전 훈련 모델 제공 (mAP@0.5 93.7%)

### 데이터셋 및 통계
- [교통안전공단 통계](https://www.ts2020.kr/) - 전동킥보드 사고 통계
- [도로교통공단 교통사고 분석](http://taas.koroad.or.kr/) - 개인형 이동장치 사고 데이터
