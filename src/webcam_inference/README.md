# 웹캠 실시간 PPE 탐지 시스템

노트북 카메라 또는 외부 웹캠을 사용한 실시간 개인보호구(PPE) 탐지 시스템

---

## 📋 개요

이 모듈은 웹캠을 통해 실시간으로 건설 현장 작업자의 안전 장비 착용 상태를 모니터링합니다.

**탐지 대상:**
- 🟦 **Helmet** (헬멧 착용) - 안전
- 🟥 **Head** (헬멧 미착용) - 위험!
- 🟨 **Vest** (안전조끼) - 안전장비

---

## 🚀 빠른 시작

### 1. 기본 실행 (노트북 내장 카메라)

```bash
uv run python src/webcam_inference/webcam_inference.py
```

### 2. 외부 웹캠 사용

```bash
uv run python src/webcam_inference/webcam_inference.py --camera 1
```

### 3. 신뢰도 임계값 조정

```bash
uv run python src/webcam_inference/webcam_inference.py --conf 0.3
```

### 4. 해상도 설정

```bash
uv run python src/webcam_inference/webcam_inference.py --width 1280 --height 720
```

---

## ⌨️ 키보드 단축키

| 키 | 기능 | 설명 |
|----|------|------|
| **Q** | 종료 | 프로그램 종료 및 리소스 해제 |
| **S** | 스크린샷 | 현재 프레임을 이미지로 저장 |
| **P** | 일시정지/재개 | 영상 일시정지 또는 재개 |
| **+** | 신뢰도 증가 | 탐지 신뢰도 임계값 +0.05 |
| **-** | 신뢰도 감소 | 탐지 신뢰도 임계값 -0.05 |
| **H** | 도움말 | 키보드 단축키 안내 표시/숨김 |

---

## 📊 화면 정보

### 통계 오버레이 (좌측 상단)

```
FPS: 30.2                          # 초당 프레임 수
Helmet: 5 | Head: 2 | Vest: 4      # 탐지된 객체 개수
Workers: 7 | Wearing Rate: 71.4%   # 작업자 수 및 헬멧 착용률
Safety: Caution                     # 안전 수준
```

### 안전 수준 평가

| 착용률 | 안전 수준 | 색상 |
|--------|----------|------|
| ≥ 90% | **Excellent** | 🟢 녹색 |
| 70-89% | **Caution** | 🟠 주황색 |
| < 70% | **Dangerous** | 🔴 빨간색 |

### 경고 메시지

헬멧 미착용자가 감지되면 화면 하단에 경고 메시지가 표시됩니다:

```
⚠️ WARNING: 2 Worker(s) Without Helmet!
```

---

## 🔧 커맨드라인 옵션

### 전체 옵션

```bash
uv run python src/webcam_inference/webcam_inference.py \
  --camera 0 \              # 카메라 인덱스 (0: 노트북, 1: 외부)
  --conf 0.25 \             # 신뢰도 임계값 (0.0-1.0)
  --width 1280 \            # 해상도 너비
  --height 720 \            # 해상도 높이
  --model path/to/model.pt \ # 커스텀 모델 경로
  --output output/screenshots  # 스크린샷 저장 디렉토리
```

### 옵션 설명

| 옵션 | 단축키 | 기본값 | 설명 |
|------|--------|--------|------|
| `--camera` | `-c` | 0 | 카메라 인덱스 (0: 내장, 1: 외부) |
| `--model` | `-m` | best.pt | YOLOv8 모델 파일 경로 |
| `--conf` | - | 0.25 | 탐지 신뢰도 임계값 (0.0-1.0) |
| `--width` | `-w` | auto | 카메라 해상도 너비 |
| `--height` | `-ht` | auto | 카메라 해상도 높이 |
| `--output` | `-o` | output/webcam_screenshots | 스크린샷 저장 경로 |

---

## 📁 파일 구조

```
src/webcam_inference/
├── __init__.py              # 패키지 초기화
├── webcam_inference.py      # 메인 실시간 추론 스크립트
├── utils.py                 # 유틸리티 함수
│   ├── FPSCounter           # FPS 계산 클래스
│   ├── calculate_statistics # 통계 계산
│   ├── draw_overlays        # 시각화 오버레이
│   └── camera utilities     # 카메라 유틸리티
└── README.md                # 이 파일
```

---

## 🎯 사용 예시

### 예시 1: 데모 모드
```bash
# 노트북 카메라로 빠르게 테스트
uv run python src/webcam_inference/webcam_inference.py
```

### 예시 2: 고해상도 모니터링
```bash
# 1080p 해상도로 실행
uv run python src/webcam_inference/webcam_inference.py \
  --width 1920 --height 1080
```

### 예시 3: 민감도 조정
```bash
# 낮은 신뢰도 (더 많이 탐지)
uv run python src/webcam_inference/webcam_inference.py --conf 0.15

# 높은 신뢰도 (확실한 것만 탐지)
uv run python src/webcam_inference/webcam_inference.py --conf 0.5
```

### 예시 4: 외부 웹캠 + 스크린샷
```bash
# 외부 웹캠 사용 및 스크린샷 저장
uv run python src/webcam_inference/webcam_inference.py \
  --camera 1 \
  --output screenshots/today

# 실행 중 'S' 키로 스크린샷 저장
```

---

## 🔍 문제 해결

### 1. 카메라를 찾을 수 없음

**증상:**
```
Error: Camera 0 is not available.
Available cameras: []
```

**해결 방법:**
1. 카메라가 다른 프로그램에서 사용 중인지 확인
2. macOS: 시스템 환경설정 > 보안 및 개인 정보 보호 > 카메라 권한 확인
3. 카메라 연결 확인 (외부 웹캠의 경우)

### 2. FPS가 낮음 (< 15 FPS)

**해결 방법:**
```bash
# 1. 해상도 낮추기
uv run python src/webcam_inference/webcam_inference.py --width 640 --height 480

# 2. 신뢰도 높이기 (처리량 감소)
uv run python src/webcam_inference/webcam_inference.py --conf 0.4

# 3. GPU 사용 (CUDA 지원 시)
# PyTorch CUDA 버전 설치 필요
```

### 3. 모델 파일을 찾을 수 없음

**증상:**
```
Error: Model file not found: models/ppe_detection/weights/best.pt
```

**해결 방법:**
```bash
# 모델 파일 존재 확인
ls models/ppe_detection/weights/best.pt

# 또는 절대 경로로 지정
uv run python src/webcam_inference/webcam_inference.py \
  --model /full/path/to/best.pt
```

### 4. 화면이 나타나지 않음

**해결 방법:**
1. OpenCV GUI 백엔드 확인
   ```bash
   python -c "import cv2; print(cv2.getBuildInformation())"
   ```
2. X11 또는 Wayland 설정 확인 (Linux)
3. 원격 SSH 접속 시 X11 포워딩 활성화

---

## ⚡ 성능 최적화

### 예상 성능

| 환경 | FPS | 추론 속도 |
|------|-----|-----------|
| **MacBook (CPU)** | 25-31 | 32-40ms |
| **Desktop (GPU)** | 50-100 | 10-20ms |
| **Raspberry Pi** | 5-10 | 100-200ms |

### 최적화 팁

1. **해상도 조정**
   - 640x480: 빠름, 낮은 품질
   - 1280x720: 균형
   - 1920x1080: 느림, 높은 품질

2. **신뢰도 임계값**
   - 0.15-0.25: 많이 탐지 (False Positive 증가)
   - 0.25-0.35: 균형 (권장)
   - 0.35-0.50: 확실한 것만 탐지

3. **GPU 활용**
   ```bash
   # CUDA 사용 가능 여부 확인
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## 📈 통계 정보

### FPS (Frames Per Second)
- 실시간 처리 속도 지표
- 30 FPS 이상: 자연스러운 영상
- 15-30 FPS: 약간 끊김
- < 15 FPS: 느림

### 헬멧 착용률
```
착용률 = (헬멧 착용자 수 / 전체 작업자 수) × 100
```

### 안전 수준
- **Excellent (우수)**: 착용률 ≥ 90%
- **Caution (주의)**: 착용률 70-89%
- **Dangerous (위험)**: 착용률 < 70%

---

## 🎓 활용 방안

### 1. 건설 현장 모니터링
- 고정 CCTV 대신 이동식 모니터링
- 작업자 안전 교육 시연

### 2. 데모 및 발표
- 실시간 탐지 시연
- 모델 성능 검증

### 3. 데이터 수집
- 스크린샷 기능으로 추가 학습 데이터 확보
- 다양한 환경에서 모델 테스트

### 4. 연구 개발
- 실시간 성능 분석
- 새로운 기능 프로토타이핑

---

## 🔗 관련 문서

- [프로젝트 메인 README](../../README.md)
- [웹캠 추론 구현 계획](../../docs/WEBCAM_INFERENCE_PLAN.md)
- [기존 추론 시스템](../inference/inference.py)
- [Streamlit 웹 인터페이스](../web_interface/app.py)

---

## 📝 참고 사항

### 개인정보 보호
- 웹캠 영상은 저장되지 않습니다 (스크린샷 제외)
- 실시간 처리만 수행됩니다
- 외부 서버로 전송되지 않습니다

### 권장 사용 환경
- **카메라**: 720p 이상 웹캠
- **CPU**: Intel i5 이상 또는 Apple M1 이상
- **RAM**: 4GB 이상
- **GPU**: 선택사항 (CUDA 지원 시 성능 향상)

### 제한 사항
- 실내 조명이 너무 어두우면 탐지 정확도 저하
- 카메라와 작업자 간 거리가 너무 멀면 탐지 어려움
- 동시에 많은 객체가 있으면 FPS 감소 가능

---

**작성자**: SafetyVisionAI Team
**최종 수정**: 2025-11-23
**버전**: 1.0.0
