# eScooterAI - 개발 진행 현황

## 📅 개발 일정

| 단계 | 기간 | 주요 목표 | 핵심 Deliverable | 상태 |
|------|------|----------|------------------|------|
| **Phase 0** | 완료 | 기반 모델 준비 | YOLOv8n 사전 훈련 모델 | ✅ 완료 |
| **Phase 1** | Week 1 (11/28~) | 프로젝트 초기화, RAG 구조 | README, RAG 시스템 | ✅ 완료 |
| **Phase 2** | Week 2-3 | **RAG 시스템 구축 (NLP 핵심)** | **벡터 DB, QA 파이프라인** | ✅ 완료 |
| **Phase 3** | Week 4 | 모바일 웹 서비스 개발 | Streamlit 멀티페이지 앱 | 📅 예정 |
| **Phase 4** | Week 5 | 통합 테스트 및 배포 | Streamlit Cloud 배포 URL | 📅 예정 |
| **Phase 5** | Week 6-7 | **실험, 논문 작성, 발표** | **학술 논문, 발표 자료** | 📅 예정 |

---

## Phase 0: 기반 모델 준비 ✅ (SafetyVisionAI)



### 주요 성과
- [v] YOLOv8n 모델 사전 훈련 완료
  - mAP@0.5: **93.7%** (목표 90% 초과 달성)
  - Precision: 92.2%, Recall: 87.2%
  - 3-class 모델: helmet, head, vest
- [v] best.pt 모델 확보
  - 파일 크기: 6.26MB (경량 모델)
  - 위치: `models/ppe_detection/weights/best.pt`
- [v] Helmet-Head 구분 정확도 검증
  - 혼동률: 0.45% (거의 완벽한 구분)
  - Helmet → Head 오분류: 29/6,304 (0.4%)
  - Head → Helmet 오분류: 6/1,024 (0.5%)
- [v] 모바일 실시간 추론 가능 확인
  - 추론 속도: 32ms/이미지
  - FPS: ~31 frames/sec

### 훈련 환경
- **GPU**: RunPod A100 (40GB)
- **훈련 시간**: 54.4분 (100 epochs)
- **Batch Size**: 128
- **Image Size**: 640×640

---

## Phase 1: 프로젝트 초기 설정 ✅



### 1.1 프로젝트 재구성 ✅
- [v] SafetyVisionAI → eScooterAI 전환
  - 건설현장 PPE 탐지 → 전동킥보드 헬멧 탐지
  - Transfer Learning 기반 도메인 전이
- [v] README.md 업데이트
  - RAG 시스템 강조
  - NLP 연구 기여점 추가
  - 프로젝트 개요 재작성

### 1.2 환경 설정 ✅
- [v] 의존성 업데이트 (pyproject.toml)
  - RAG 라이브러리 추가
    - `faiss-cpu>=1.7.4` (벡터 검색)
    - `openai>=1.0.0` (LLM API)
    - `tiktoken>=0.5.0` (토큰 카운팅)
  - 불필요한 라이브러리 제거
    - TensorFlow, Jupyter, gTTS, pygame 제거
- [v] 환경 변수 설정
  - `.env.example` 파일 생성
  - OpenAI API, Telegram Bot 설정 추가

### 1.3 프로젝트 구조 재구성 ✅
- [v] `src/rag/` 디렉토리 생성
  - RAG 시스템 모듈화
- [v] `src/data/safety_docs/` 디렉토리 생성
  - 안전 교육 지식 베이스 저장소
- [v] `src/mobile_app/` 디렉토리 생성
  - Streamlit 모바일 앱 구조 (향후 구현)

---

## Phase 2: RAG 시스템 구축 ✅


### 2.1 안전 문서 데이터 수집 및 전처리 ✅

#### 법규 문서 (10개) ✅
- [v] 도로교통법 제50조 제3항 (헬멧 착용 의무)
- [v] 도로교통법 제160조 (과태료 2만원)
- [v] 도로교통법 제13조의2 (통행 방법)
- [v] 도로교통법 제44조 (음주운전 금지)
- [v] 도로교통법 시행규칙 제35조 (속도 제한)
- [v] 연령 제한, 인도 주행, 2인 탑승 금지 등

#### 안전 가이드 (12개) ✅
- [v] 헬멧 올바른 착용법
- [v] 헬멧 선택 요령 (KC 인증)
- [v] 전동킥보드 안전 운전 수칙
- [v] 헬멧 관리 방법
- [v] 탑승 자세, 야간 운행, 응급처치 등

#### 사고 사례 및 통계 (13개) ✅
- [v] 2023년 사망사고 통계 (헬멧 미착용 92%)
- [v] 실제 사고 사례 (서울, 부산, 인천 등)
- [v] 헬멧 착용률 통계 (전국 38.5%)
- [v] 음주운전 적발 사례
- [v] 야간 사고 통계 (62% 야간 발생)

#### 문서 전처리 ✅
- [v] JSON 형식으로 구조화
  - `src/data/safety_docs/laws.json`
  - `src/data/safety_docs/guides.json`
  - `src/data/safety_docs/cases.json`
- [v] 메타데이터 태깅
  - category, source, year, law_type 등
- [v] 총 35개 문서 구축 완료

### 2.2 벡터 DB 구축 (FAISS) ✅

#### 임베딩 모델 선택 ✅
- [v] OpenAI text-embedding-3-small 선택
  - 이유: 높은 성능, 1536차원 밀집 벡터
  - 비용: $0.00002 / 1K tokens (매우 저렴)
  - 속도: 빠름

#### FAISS 인덱스 설정 ✅
- [v] IndexFlatL2 생성 (L2 거리 기반)
  - Exact Nearest Neighbor Search (정확도 100%)
  - 35개 문서로 충분히 빠름
- [v] 벡터 저장소 구현 완료
  - `src/rag/vector_store.py` (282줄)
  - save/load 기능으로 영구 저장 가능

#### 주요 기능 ✅
- [v] `get_embedding()`: 텍스트 → 1536차원 벡터
- [v] `add_documents()`: 문서 임베딩 및 인덱스 추가
- [v] `search()`: Top-K 유사도 검색
- [v] `save()` / `load()`: 디스크 저장/로드

### 2.3 RAG 파이프라인 구현 (직접 구현) ✅

#### Retriever 구현 ✅
- [v] Query Embedding 생성 (OpenAI API)
- [v] L2 Distance 기반 Top-K 검색 (FAISS)
- [v] 검색 결과 필터링 및 정렬
- [v] 거리 → 유사도 점수 변환 (score = 1/(1+distance))

#### Generator 구현 ✅
- [v] OpenAI API 통합
  - GPT-4 Turbo / GPT-3.5 Turbo 지원
  - `src/rag/query_engine.py` (217줄)
- [v] Context 구성
  - 검색된 문서 + 사용자 질문
  - 출처 정보 포함
- [v] Temperature, Max Tokens 설정
  - Temperature: 0.3 (사실 기반 답변)
  - Max Tokens: 500

#### Prompt Engineering ✅
- [v] System Prompt: "전동킥보드 안전 교육 전문가"
- [v] 5가지 지침 제시
  1. 참고 문서 기반 정확한 답변
  2. 법규는 조항과 벌금 명시
  3. 안전 가이드는 구체적 방법 설명
  4. 없는 정보는 "자료에 없음" 답변
  5. 2-3문장 간결성
- [v] Context grounding (환각 방지)

#### RAG 파이프라인 통합 ✅
- [v] 3단계 파이프라인 구현
  1. **Retrieval**: FAISS 벡터 검색
  2. **Augmentation**: 프롬프트 구성
  3. **Generation**: LLM 답변 생성
- [v] 에러 핸들링
  - 검색 실패 시 안내 메시지
  - API 오류 처리

### 2.4 RAG 평가 및 테스트 시스템 구축 ✅

#### 평가 스크립트 구현 ✅
- [v] `src/rag/test_rag.py` (251줄)
- [v] `src/rag/build_vector_db.py` (179줄)

#### 검색 성능 평가 ✅
- [v] Precision@K 구현
  - Top-3 중 올바른 카테고리 포함 여부
  - 테스트 케이스 9개 준비
- [v] Recall@K 준비 (향후 측정)

#### 답변 품질 평가 ✅
- [v] Response Time 측정
- [v] Token Usage 측정
- [v] Answer Relevance (수동 평가 준비)
- [v] Hallucination Check (엣지 케이스 테스트)

#### 엣지 케이스 테스트 ✅
- [v] Missing Info (환각 방지)
  - "전동킥보드 보험료는 얼마야?" → "자료에 없음" 답변 확인
- [v] Ambiguous Queries (모호한 질문)
  - "안전하게 타려면?" → 일반 안전 수칙 답변
- [v] Complex Queries (복합 질문)
  - "헬멧 안 쓰고 인도로 달리면?" → 다중 벌금 합산

### 2.5 코드 주석 및 문서화 ✅
- [v] 주요 RAG 모듈에 상세 주석 추가
  - 학술 논문 작성에 유용한 기술적 설명
  - 알고리즘 복잡도, 수식, 데이터 구조 명시
  - NLP/ML 표준 용어 사용
- [v] 주석 추가 파일
  - `vector_store.py`: FAISS 인덱스, L2 거리, 임베딩 설명
  - `query_engine.py`: RAG 파이프라인, 프롬프트 엔지니어링
  - `build_vector_db.py`: 벡터 DB 구축 프로세스
  - `test_rag.py`: 평가 지표 (Precision@K, Hallucination)

---

## Phase 3: 모바일 웹 서비스 구현 📱 (예정)

**예정일**: Week 4

### 3.1 멀티페이지 Streamlit 앱 구조
- [ ] 페이지 1: 실시간 헬멧 탐지 (WebRTC)
- [ ] 페이지 2: 안전 가이드 챗봇 (RAG)
- [ ] 홈 페이지: 프로젝트 소개 및 사용법

### 3.2 WebRTC 기반 실시간 탐지
- [ ] streamlit-webrtc 통합
- [ ] 모바일 카메라 스트리밍
- [ ] 프레임 단위 YOLO 추론
- [ ] 실시간 바운딩 박스 오버레이

### 3.3 Telegram 알림 통합
- [ ] 헬멧 미착용 감지 시 즉각 알림
- [ ] 탐지 결과 이미지 전송
- [ ] 알림 설정 UI (사용자가 활성화/비활성화)

### 3.4 모바일 UI/UX 최적화
- [ ] 반응형 레이아웃 (모바일/태블릿/데스크톱)
- [ ] 터치 인터페이스 최적화
- [ ] 로딩 시간 최소화

---

## Phase 4: 통합 테스트 및 배포 🚀 (예정)

**예정일**: Week 5

### 4.1 로컬 테스트
- [ ] 헬멧 탐지 기능 테스트 (다양한 각도, 조명)
- [ ] RAG 챗봇 응답 품질 평가
- [ ] Telegram 알림 테스트
- [ ] 모바일 브라우저 호환성 테스트

### 4.2 Streamlit Community Cloud 배포
- [ ] GitHub 연동 자동 배포 설정
- [ ] 환경 변수 및 Secrets 설정 (OpenAI API, Telegram Bot)
- [ ] 배포 URL 확보
- [ ] 배포 가이드 문서 작성

### 4.3 성능 최적화
- [ ] 모델 추론 속도 최적화
- [ ] 벡터 DB 쿼리 속도 최적화
- [ ] 메모리 사용량 최적화

---

## Phase 5: 논문 작성 및 발표 📝 (예정)

**예정일**: Week 6-7

### 5.1 실험 결과 정리

#### 컴퓨터 비전 실험
- [ ] 헬멧 탐지 정확도 측정 (전동킥보드 환경)
  - [ ] 실제 전동킥보드 이용자 테스트 영상 수집
  - [ ] 다양한 조명/각도/속도에서 탐지 성능
  - [ ] mAP@0.5, Precision, Recall 측정
  - [ ] Transfer Learning 효과 분석

#### 자연어 처리 실험 (핵심) 🔬
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

#### 통합 시스템 평가
- [ ] End-to-End 성능 측정 (탐지 → 알림 → 챗봇)
- [ ] 기존 시스템 대비 우수성 입증
- [ ] 사용 시나리오별 효과성 분석

### 5.2 학술 논문 작성

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
  - [ ] RAG 시스템 (FAISS, OpenAI)
  - [ ] Domain-Specific QA Systems
  - [ ] Transfer Learning in Safety Domain

- [ ] **3. Methodology (방법론)**
  - [ ] 3.1 System Architecture (전체 아키텍처)
  - [ ] 3.2 Helmet Detection Module (YOLOv8n)
  - [ ] 3.3 **RAG-based QA Module (핵심)**
    - [ ] Document Collection & Preprocessing
    - [ ] Vector Embedding (FAISS, OpenAI)
    - [ ] Semantic Retrieval (L2 Distance)
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

### 5.3 최종 발표 준비
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

## 완료된 개선사항 (과거 프로젝트)

### SafetyVisionAI 기반 작업
- ✅ **헬멧 미착용 탐지**: head 클래스 추가로 헬멧 미착용 상태 감지 가능
- ✅ **데이터 재구성**: 3 class (helmet, head, vest) 데이터셋 구축 완료
- ✅ **3 Class 모델 성능 검증 완료**
  - A100 GPU 100 epochs 본 훈련 완료
  - 목표 성능 달성: **mAP@0.5 = 93.7%** (목표 90% 초과)
  - Helmet-Head 혼동률 0.45%로 거의 완벽한 구분

- ✅ **통합 추론 시스템 구현 완료**
  - helmet, head, vest 3개 클래스 동시 탐지
  - 단일 이미지 / 디렉토리 처리 지원
  - 헬멧 착용률 자동 계산 및 안전 수준 평가
  - 시각화 결과 (PNG) 및 JSON 저장
  - 명령줄 인터페이스 (CLI) 지원

- ✅ **웹 인터페이스 구축 완료**
  - Streamlit 기반 대시보드 개발 완료
  - 이미지 업로드 및 실시간 탐지 결과 표시
  - 바운딩 박스 시각화 및 원본/결과 비교 뷰
  - 헬멧 착용률 및 안전 수준 평가 자동화
  - 신뢰도/IoU 임계값 설정 UI

- ✅ **Streamlit Community Cloud 배포 완료** 
  - GitHub 연동 자동 배포
  - YOLOv8 모델 포함 (best.pt, 6.0MB)
  - 무료 호스팅 (Streamlit Community Cloud)
  - 실시간 웹 데모: https://escooterai.streamlit.app

- ✅ **Telegram Bot 실시간 알림 시스템 구축 완료**  📱
  - 헬멧 미착용자 2명 이상 또는 착용률 80% 미만 시 자동 알림
  - Telegram 그룹 채팅 지원 (팀원 모두 알림 수신)
  - 탐지 결과 이미지 포함 전송
  - 안전 수준별 메시지 포맷 (Excellent/Caution/Dangerous)

---

## 향후 과제 및 개선 계획

### 🎯 핵심 개발 과제 (Phase 3-4)

#### 1. 모바일 웹 서비스 개발 (최우선)
- Streamlit 멀티페이지 구조 설계
- WebRTC 기반 실시간 카메라 스트리밍
- 모바일 반응형 UI/UX 최적화

#### 2. RAG 챗봇 UI 통합
- RAG 쿼리 엔진 UI 개발
- 질의응답 인터페이스
- 출처 문서 표시 기능

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
- Retrieval Precision@K 측정
- Answer Relevance, Hallucination Rate 측정
- 사용자 만족도 조사 (설문/인터뷰)
- 프롬프트 엔지니어링 최적화

---

## 📊 주요 성과 요약

### Phase 0-1 (기반 구축)
- YOLOv8n 모델: mAP@0.5 93.7%
- 프로젝트 구조 재설계 완료
- 환경 설정 및 의존성 관리

### Phase 2 (RAG 시스템) 🔬
- **안전 교육 지식 베이스**: 35개 문서 (법규 10, 가이드 12, 사례 13)
- **벡터 저장소**: FAISS IndexFlatL2, OpenAI 임베딩 (1536차원)
- **RAG 파이프라인**: Retrieval → Augmentation → Generation 완전 구현
- **평가 시스템**: Precision@K, Hallucination Check, Response Time 측정 준비
- **상세 주석**: 학술 논문 작성용 기술 문서화 완료

### 다음 목표 (Phase 3-5)
- 모바일 웹 서비스 개발 및 배포
- NLP 연구 실험 및 평가
- 학술 논문 작성 및 발표

---
