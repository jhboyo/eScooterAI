# Phase 7: 웹 인터페이스 구축

## 📋 목표

Streamlit 기반의 직관적인 웹 대시보드를 구축하여 PPE Detection 시스템을 누구나 쉽게 사용할 수 있도록 합니다.

---

## 📊 진행 상황

### ✅ Step 1: 기본 구조 설정 (완료)

#### 구현 내용
1. **프로젝트 구조 생성**
   ```
   src/web_interface/
   ├── __init__.py
   ├── app.py                    # 메인 Streamlit 앱
   ├── components/
   │   └── __init__.py
   ├── utils/
   │   └── __init__.py
   └── assets/                   # 정적 파일용
   ```

2. **의존성 설치**
   - ✅ streamlit==1.51.0
   - ✅ plotly==6.5.0
   - ✅ altair==5.3.0

3. **메인 앱 구현 (app.py)**
   - 페이지 설정 (타이틀, 아이콘, 레이아웃)
   - 커스텀 CSS 스타일링
   - 사이드바 설정 패널
     - 모델 선택 (best.pt / last.pt)
     - 신뢰도 임계값 슬라이더 (0.1-1.0)
     - 고급 옵션 (IoU, Max Detections)
   - 파일 업로더 UI (다중 파일 지원)
   - 업로드된 파일 정보 표시
   - Placeholder 섹션 (결과, 통계)

4. **테마 설정 (.streamlit/config.toml)**
   - Primary Color: `#1f4068`
   - 서버 포트: 8501
   - 최적화 설정 적용

5. **테스트 완료**
   - ✅ 모듈 import 성공
   - ✅ 의존성 정상 작동
   - ✅ 기본 UI 렌더링 검증

#### 실행 방법
```bash
# 웹 대시보드 실행
uv run streamlit run src/web_interface/app.py

# 브라우저 자동 접속
# 수동 접속: http://localhost:8501
```

#### 구현된 UI
- 🏗️ 메인 헤더 (Safety Vision AI)
- 📁 이미지 업로드 섹션 (드래그 앤 드롭)
- ⚙️ 사이드바 설정 패널
- ℹ️ 정보 패널 (클래스, 안전 수준)
- 📊 결과 표시 영역 (Placeholder)

#### 다음 단계
- Step 2: 이미지 업로드 기능 구현
  - 이미지 미리보기
  - 파일 검증 강화
  - 세션 상태 관리

---

## 🎯 주요 기능

### 1. 이미지 업로드 및 추론
- 드래그 앤 드롭 방식 이미지 업로드
- 다중 이미지 업로드 지원
- 지원 형식: JPG, PNG, WEBP

### 2. 실시간 탐지 결과 표시
- 원본 이미지와 탐지 결과 나란히 비교
- 바운딩 박스 시각화 (helmet: 파란색, head: 빨간색, vest: 노란색)
- 클래스별 신뢰도 표시

### 3. 통계 및 분석
- 헬멧 착용률 자동 계산
- 안전 수준 평가 (Excellent / Caution / Dangerous)
- 클래스별 탐지 개수 차트
- 이미지별 통계 테이블

### 4. 대시보드
- 실시간 안전 지표 모니터링
- 히스토리 관리 (세션 내)
- 결과 다운로드 (이미지 + JSON)

---

## 🛠️ 기술 스택

| 구분 | 기술 |
|------|------|
| **웹 프레임워크** | Streamlit |
| **데이터 시각화** | Plotly, Matplotlib |
| **이미지 처리** | PIL, OpenCV |
| **모델 추론** | Ultralytics YOLOv8 |
| **상태 관리** | Streamlit Session State |

---

## 📂 파일 구조

```
src/
└── web_interface/
    ├── __init__.py
    ├── app.py                  # 메인 Streamlit 앱
    ├── components/             # UI 컴포넌트
    │   ├── __init__.py
    │   ├── uploader.py        # 이미지 업로드 컴포넌트
    │   ├── detector.py        # 탐지 실행 컴포넌트
    │   ├── visualizer.py      # 결과 시각화 컴포넌트
    │   └── statistics.py      # 통계 차트 컴포넌트
    ├── utils/                  # 유틸리티 함수
    │   ├── __init__.py
    │   ├── inference.py       # 추론 로직
    │   └── plotting.py        # 차트 생성
    └── assets/                 # 정적 파일
        ├── logo.png
        └── styles.css
```

---

## 🎨 UI/UX 설계

### 레이아웃 구조

```
┌─────────────────────────────────────────────────┐
│  🏗️ Safety Vision AI - PPE Detection Dashboard  │
├─────────────────────────────────────────────────┤
│  📁 File Upload Section                          │
│  [Drag & Drop or Browse Files]                  │
│                                                  │
│  ⚙️ Settings                                     │
│  [Confidence Threshold: 0.25]                   │
│  [Model: best.pt]                               │
│  [Process Images Button]                        │
├─────────────────────────────────────────────────┤
│  📊 Results Section                              │
│  ┌─────────────┬─────────────┐                  │
│  │ Original    │ Detection   │                  │
│  │ Image       │ Result      │                  │
│  └─────────────┴─────────────┘                  │
│                                                  │
│  📈 Statistics                                   │
│  - Helmet: 5 (100%)                             │
│  - Head: 0 (0%)                                 │
│  - Vest: 4                                      │
│  - Safety Level: ✅ Excellent                    │
│                  │
│        │
├─────────────────────────────────────────────────┤
```

---

## 📝 구현 단계

### Step 1: 기본 구조 설정 ✅ 완료
- [x] Streamlit 프로젝트 초기화
- [x] 기본 레이아웃 구성
- [x] 네비게이션 구조 설계
- [x] 테마 및 스타일 적용

### Step 2: 이미지 업로드 및 미리보기 ✅ 완료

#### 목표
업로드된 이미지를 검증하고 미리보기를 제공하여 사용자가 올바른 파일을 선택했는지 확인

#### 세부 작업
1. **파일 업로더 컴포넌트 구현**
   - [x] `components/uploader.py` 생성
   - [x] `st.file_uploader()` 설정 (다중 파일, 드래그앤드롭)
   - [x] 지원 형식: JPG, JPEG, PNG, WEBP, BMP
   - [x] 최대 파일 크기 제한: 10MB per file

2. **이미지 검증**
   - [x] 파일 형식 검증 (확장자 확인)
   - [x] 파일 크기 검증 (10MB 제한)
   - [x] 손상된 이미지 감지 (PIL.Image.verify())
   - [x] 에러 메시지 표시 (한글 메시지)

3. **이미지 미리보기**
   - [x] 그리드 레이아웃으로 썸네일 표시 (3열 그리드)
   - [x] 파일명, 크기, 해상도 정보 표시
   - [x] 개별 이미지 삭제 버튼 (세션 상태 업데이트)
   - [x] 전체 이미지 개수 및 총 용량 표시

4. **세션 상태 관리**
   - [x] `st.session_state.uploaded_files` 초기화
   - [x] 업로드된 이미지를 메모리에 저장 (PIL.Image)
   - [x] 이미지 메타데이터 저장 (파일명, 크기, 해상도, 형식)

5. **코드 품질 개선** (추가 작업)
   - [x] 풍선 효과 제거 (st.balloons() → st.info())
   - [x] CSS 파일 분리 (assets/styles.css)
   - [x] 한글 주석 추가 (모든 함수 및 주요 로직)
   - [x] Sidebar 로고 심플화

#### 테스트 항목
- [x] 단일 이미지 업로드 정상 작동
- [x] 다중 이미지 업로드 정상 작동
- [x] 잘못된 형식 파일 업로드 시 에러 표시
- [x] 크기 초과 파일 업로드 시 에러 표시
- [x] 미리보기 그리드 정상 표시
- [x] 개별 삭제 버튼 작동
- [x] 전체 삭제 버튼 작동
- [x] CSS 파일 로드 정상 작동
- [x] Python 모듈 import 정상 작동

#### 구현 파일
- `src/web_interface/components/uploader.py` (313줄)
- `src/web_interface/assets/styles.css` (197줄)
- `src/web_interface/app.py` (업데이트)

#### 다음 단계
Step 3에서 업로드된 이미지를 YOLOv8 모델에 전달하여 PPE 탐지 추론 수행

---

### Step 3: 모델 통합 및 추론 ✅ 완료

#### 목표
YOLOv8 모델을 로드하고 업로드된 이미지에 대해 배치 추론을 수행

#### 세부 작업
1. **모델 로드 최적화**
   - [x] `utils/inference.py` 생성
   - [x] `@st.cache_resource`로 모델 캐싱
   - [x] 모델 경로 검증 (파일 존재 확인)
   - [x] 모델 로드 실패 시 에러 처리
   - [x] 로딩 스피너 표시

2. **추론 파이프라인 구현**
   - [x] 단일 이미지 추론 함수 작성
   - [x] 배치 이미지 추론 함수 작성
   - [x] 설정값 적용 (conf, iou, max_det)
   - [x] 추론 결과 파싱 (boxes, classes, confidences)

3. **진행 상태 표시**
   - [x] `st.progress()` 구현
   - [x] 현재 처리 중인 이미지 표시
   - [x] 예상 완료 시간 계산
   - [x] 추론 속도 (FPS) 표시

4. **에러 처리**
   - [x] 모델 로드 실패 처리
   - [x] 추론 중 예외 처리
   - [x] GPU/CPU 자동 전환
   - [x] 타임아웃 설정 (이미지당 최대 30초)



#### 테스트 항목
- [x] 모델 로드 정상 작동
- [x] 캐싱으로 재로드 시 즉시 로드
- [x] 단일 이미지 추론 정상
- [x] 다중 이미지 배치 추론 정상
- [x] 진행 상태 바 정상 작동
- [x] 예상 완료 시간 표시
- [x] FPS 계산 정확
- [x] 에러 발생 시 적절한 메시지

#### 다음 단계
Step 4에서 추론 결과를 시각화하여 사용자에게 표시

---

### Step 4: 결과 시각화 ✅ 완료

#### 목표
추론 결과를 바운딩 박스와 함께 시각화하고 원본/결과 비교 뷰 제공

#### 세부 작업
1. **바운딩 박스 그리기**
   - [x] `utils/plotting.py` 생성
   - [x] PIL.ImageDraw를 사용한 바운딩 박스 그리기
   - [x] 클래스별 색상 정의 (helmet: 파랑, head: 빨강, vest: 노랑)
   - [x] 선 두께 조정 (이미지 크기에 비례)
   - [x] 신뢰도 라벨 텍스트 추가

2. **원본/결과 비교 표시**
   - [x] 2열 레이아웃 구성 (원본 | 결과)
   - [x] 이미지 확대/축소 기능
   - [x] 이미지별 네비게이션 (이전/다음)
   - [x] 탭 또는 슬라이더로 이미지 전환

3. **클래스별 색상 및 스타일**
   - [x] 색상 사전 정의
   - [x] 폰트 설정 (크기, 스타일)
   - [x] 반투명 라벨 배경
   - [x] 안티앨리어싱 적용

4. **상세 정보 표시**
   - [x] 탐지된 객체 수 표시
   - [x] 각 탐지 객체의 신뢰도 리스트
   - [x] 바운딩 박스 좌표 표시 (선택사항)

5. **디버그 모드 추가** (추가 작업)
   - [x] 고급 옵션에 디버그 모드 체크박스
   - [x] 클래스별 상세 탐지 정보 표시
   - [x] Head 클래스 필터링 뷰

#### 테스트 항목
- [x] 바운딩 박스 정확하게 그려짐
- [x] 클래스별 색상 올바르게 적용
- [x] 신뢰도 라벨 표시
- [x] 원본/결과 비교 뷰 정상 작동
- [x] 이미지 슬라이더 작동
- [x] 탐지 상세 정보 표시
- [x] 다양한 해상도 이미지 대응

#### 발견된 중요 이슈
⚠️ **DS2 데이터셋 Head 클래스 탐지 실패 (심각)**
- DS2 스타일 이미지(어두운 배경, 검은색 머리)에서 Head 클래스 탐지 0%
- 근본 원인: DS2 데이터셋에 Head 레이블 없음 (DS1만 있음)
- 안전 위험: 헬멧 미착용자를 착용자로 오인 가능
- 자세한 내용: README.md "🚨 현재 발견된 문제점 및 개선 과제" 참고

#### 다음 단계
Step 4에서 추론 결과를 시각화하여 사용자에게 표시

---

## 🎬 주요 기능 상세 설명

### 1. 이미지 업로드 (`components/uploader.py`)
- 드래그 앤 드롭 방식 이미지 업로드
- 다중 파일 지원
- 파일 검증 및 에러 처리

### 2. 탐지 실행 (`components/detector.py`)
- YOLOv8 모델 로드 (캐싱)
- 배치 이미지 추론
- 진행 상태 표시

### 3. 통계 계산 (`components/statistics.py`)
- 탐지 결과 통계 계산
- 헬멧 착용률 계산
- 안전 수준 평가
- 차트 생성

---

## 🎨 UI 컴포넌트

### 메트릭 카드
- st.metric()을 사용한 KPI 표시
- 4열 레이아웃 구성
- 총 작업자, 헬멧 착용, 미착용, 안전조끼 수 표시

### 안전 수준 표시
- 헬멧 착용률에 따른 3단계 평가
- Excellent (90% 이상), Caution (70-90%), Dangerous (70% 미만)
- 색상 코드로 시각적 구분

---

## ⚙️ 설정 옵션

### 사이드바 설정
- 모델 선택 (best.pt / last.pt)
- 신뢰도 임계값 슬라이더 (0.1-1.0)
- 고급 옵션
  - IoU 임계값
  - 최대 탐지 개수

---

## 🚀 실행 방법

### 1. 의존성 설치
- `uv add streamlit plotly` 명령으로 설치

### 2. 앱 실행
- 개발 모드: `uv run streamlit run src/web_interface/app.py`
- 프로덕션 모드: 포트 지정 옵션 사용 가능

### 3. 브라우저 접속
- 기본 주소: http://localhost:8501

---

## 🎯 성공 기준

### 기능적 요구사항
- [x] 이미지 업로드 및 다중 파일 처리
- [x] 실시간 추론 및 결과 표시
- [x] 헬멧 착용률 자동 계산
- [x] 안전 수준 3단계 평가
- [ ] 결과 다운로드 기능 (선택사항)

### 비기능적 요구사항
- [ ] 첫 로딩 시간: < 5초
- [ ] 이미지 1장 추론: < 1초
- [ ] 반응형 UI (모바일 지원)
- [ ] 에러 메시지 명확성
- [ ] 직관적인 사용자 경험

---

## 📚 참고 자료

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python](https://plotly.com/python/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [YOLOv8 Streamlit Example](https://github.com/ultralytics/ultralytics/tree/main/examples/streamlit)

---

## ✅ 체크리스트

### 개발 준비
- [x] Streamlit 설치 및 테스트
- [x] 프로젝트 구조 생성

### 개발
- [x] Step 1: 기본 구조 설정
- [x] Step 2: 이미지 업로드
- [x] Step 3: 모델 통합
- [x] Step 4: 결과 시각화

### 배포
- [ ] 로컬 테스트 완료
- [ ] 문서 작성
- [ ] README 업데이트
- [ ] 데모 비디오 녹화

---



