# 웹캠 실시간 추론 구현 계획

## 📋 프로젝트 개요

**목표**: 노트북 내장 카메라를 사용한 실시간 PPE(개인보호구) 탐지 시스템 구현

**Phase**: Phase 8 - 실시간 추론 및 성능 개선

**작성일**: 2025-11-23

---

## 🎯 구현 목표

### 주요 기능

1. **실시간 영상 처리**
   - 노트북 내장 카메라 (또는 외부 USB 웹캠) 스트림 캡처
   - 실시간 프레임 단위 객체 탐지 (helmet, head, vest)

2. **실시간 시각화**
   - 바운딩 박스 실시간 표시 (클래스별 색상 구분)
   - 탐지 통계 오버레이 (헬멧/머리/조끼 개수)
   - FPS (Frames Per Second) 카운터 표시

3. **안전 모니터링**
   - 헬멧 착용률 실시간 계산
   - 안전 수준 평가 (Excellent/Caution/Dangerous)
   - 헬멧 미착용자 감지 시 경고 표시

4. **사용자 제어**
   - 키보드 단축키로 제어
   - 스크린샷 저장 기능
   - 프로그램 종료 기능

---

## 🏗️ 기술 스택

| 항목 | 기술 | 버전 | 용도 |
|------|------|------|------|
| **카메라 캡처** | OpenCV | 4.8.0+ | 웹캠 스트림 처리 |
| **객체 탐지** | YOLOv8n | - | PPE 실시간 탐지 |
| **모델** | best.pt | - | 학습된 가중치 (mAP 93.7%) |
| **시각화** | OpenCV | 4.8.0+ | 바운딩 박스, 텍스트 오버레이 |

---

## 📁 프로젝트 구조

```
src/webcam_inference/
├── __init__.py                 # 패키지 초기화
├── webcam_inference.py         # 메인 스크립트 (실시간 추론)
├── utils.py                    # 유틸리티 함수 (통계 계산, FPS 측정)
└── README.md                   # 사용 가이드
```

---

## 🔧 핵심 구현 로직

### 1. 카메라 초기화
```python
import cv2

# 노트북 내장 카메라 (0), 외부 웹캠 (1)
cap = cv2.VideoCapture(0)

# 해상도 설정 (선택사항)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

### 2. 모델 로드
```python
from ultralytics import YOLO

model = YOLO('models/ppe_detection/weights/best.pt')
```

### 3. 실시간 추론 루프
```python
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 추론
    results = model(frame, conf=0.25)

    # 결과 시각화
    annotated_frame = results[0].plot()

    # 통계 오버레이 추가
    annotated_frame = add_statistics(annotated_frame, results)

    # 화면 표시
    cv2.imshow('PPE Detection - Real-time', annotated_frame)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 종료
        break
    elif key == ord('s'):  # 스크린샷
        save_screenshot(annotated_frame)
```

### 4. FPS 계산
```python
import time

fps_counter = []
start_time = time.time()

# 프레임 처리 시작
process_start = time.time()

# ... 추론 및 시각화 ...

# FPS 계산
process_time = time.time() - process_start
fps = 1.0 / process_time if process_time > 0 else 0
fps_counter.append(fps)

# 평균 FPS (최근 30프레임)
avg_fps = sum(fps_counter[-30:]) / len(fps_counter[-30:])
```

---

## 📊 예상 성능

| 지표 | 예상 값 | 근거 |
|------|---------|------|
| **모델 추론 속도** | 32ms/프레임 | 기존 테스트 결과 |
| **FPS** | 25-31 FPS | 1000ms / 32ms ≈ 31 FPS |
| **실시간 처리** | ✅ 가능 | 30 FPS 이상이면 자연스러움 |
| **GPU 사용 시** | 50+ FPS | 추론 속도 10-15ms 예상 |

**결론**: YOLOv8n (Nano) 경량 모델로 충분히 실시간 처리 가능

---

## 🎨 UI 디자인

### 화면 구성

```
┌─────────────────────────────────────────────────┐
│  [웹캠 영상]                                      │
│  ┌──────────────────────────────────────┐       │
│  │                                        │       │
│  │   🟦 Helmet (파란색 박스)             │       │
│  │   🟥 Head (빨간색 박스) ⚠️            │       │
│  │   🟨 Vest (노란색 박스)               │       │
│  │                                        │       │
│  └──────────────────────────────────────┘       │
│                                                   │
│  [통계 정보 오버레이]                             │
│  ┌──────────────────────────────────────┐       │
│  │ FPS: 30.2                             │       │
│  │ Helmet: 5 | Head: 2 ⚠️ | Vest: 4    │       │
│  │ Wearing Rate: 71.4%                   │       │
│  │ Safety Level: ⚠️ CAUTION             │       │
│  └──────────────────────────────────────┘       │
│                                                   │
│  [조작 안내]                                      │
│  Q: 종료 | S: 스크린샷 | C: 카메라 전환          │
└─────────────────────────────────────────────────┘
```

### 색상 코드

| 요소 | 색상 (BGR) | 의미 |
|------|-----------|------|
| Helmet 박스 | (255, 0, 0) | 파란색 - 안전 |
| Head 박스 | (0, 0, 255) | 빨간색 - 위험 |
| Vest 박스 | (0, 255, 255) | 노란색 - 안전장비 |
| 통계 배경 | (0, 0, 0) | 검정색 (반투명) |
| 통계 텍스트 | (255, 255, 255) | 흰색 |
| 경고 텍스트 | (0, 0, 255) | 빨간색 |

---

## ⌨️ 키보드 컨트롤

| 키 | 기능 | 설명 |
|----|------|------|
| **Q** | 종료 | 프로그램 종료 및 리소스 해제 |
| **S** | 스크린샷 | 현재 프레임 저장 (`output/webcam_screenshots/`) |
| **C** | 카메라 전환 | 여러 카메라 간 전환 (0, 1, 2, ...) |
| **P** | 일시정지 | 영상 일시정지/재개 |
| **+/-** | 신뢰도 조정 | 탐지 신뢰도 임계값 증가/감소 |
| **H** | 도움말 | 키보드 단축키 안내 표시 |

---

## 🚀 실행 방법

### 1. 기본 실행 (노트북 카메라)
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

### 5. 커스텀 모델 사용
```bash
uv run python src/webcam_inference/webcam_inference.py --model path/to/model.pt
```

---

## 📋 구현 체크리스트

### Phase 1: 기본 구현 (필수)
- [ ] 카메라 초기화 및 스트림 캡처
- [ ] YOLOv8 모델 로드
- [ ] 실시간 추론 루프 구현
- [ ] 바운딩 박스 시각화
- [ ] 기본 통계 오버레이 (helmet, head, vest 개수)
- [ ] 종료 키 (Q) 구현

### Phase 2: 통계 및 경고 (중요)
- [ ] FPS 계산 및 표시
- [ ] 헬멧 착용률 계산
- [ ] 안전 수준 평가 (Excellent/Caution/Dangerous)
- [ ] 헬멧 미착용자 감지 시 경고 표시
- [ ] 통계 정보 오버레이 디자인

### Phase 3: 사용자 제어 (개선)
- [ ] 스크린샷 저장 (S 키)
- [ ] 카메라 전환 (C 키)
- [ ] 일시정지/재개 (P 키)
- [ ] 신뢰도 임계값 조정 (+/- 키)
- [ ] 도움말 표시 (H 키)

### Phase 4: 고급 기능 (선택)
- [ ] 녹화 기능 (영상 저장)
- [ ] Telegram Bot 실시간 알림 연동
- [ ] 경고음 (비프음) 재생
- [ ] 탐지 히스토리 차트
- [ ] 다중 카메라 동시 모니터링

---

## 🐛 예상 문제 및 해결 방안

### 1. 카메라 접근 권한 오류
**문제**: macOS에서 카메라 접근 권한이 없음
```
[ERROR] Can't open camera
```

**해결**:
```bash
# 시스템 환경설정 > 보안 및 개인 정보 보호 > 카메라
# 터미널 또는 Python에 카메라 접근 권한 부여
```

### 2. 낮은 FPS
**문제**: 실시간 처리 속도가 느림 (< 15 FPS)

**해결**:
- 입력 해상도 낮추기 (640x480)
- 신뢰도 임계값 높이기 (0.4-0.5)
- GPU 사용 (CUDA 지원 PyTorch 설치)
- 배치 추론 비활성화 (단일 프레임)

### 3. 모델 로드 실패
**문제**: 모델 파일을 찾을 수 없음

**해결**:
```bash
# 모델 경로 확인
ls models/ppe_detection/weights/best.pt

# 절대 경로로 지정
python webcam_inference.py --model /full/path/to/best.pt
```

### 4. 메모리 누수
**문제**: 장시간 실행 시 메모리 사용량 증가

**해결**:
```python
# 프레임 버퍼 제한
if len(frame_buffer) > 100:
    frame_buffer.clear()

# OpenCV 리소스 해제
cv2.destroyAllWindows()
cap.release()
```

---

## 📈 성능 최적화 전략

### 1. 프레임 스킵 (Frame Skip)
```python
frame_count = 0
SKIP_FRAMES = 2  # 2프레임마다 1번 추론

if frame_count % SKIP_FRAMES == 0:
    results = model(frame, conf=0.25)
frame_count += 1
```
**효과**: FPS 2-3배 증가, 약간의 반응 지연

### 2. 해상도 조정
```python
# 고해상도 → 저해상도
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 1280 → 640
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 720 → 480
```
**효과**: 추론 속도 40-50% 향상

### 3. GPU 활용
```python
# GPU 가용성 확인
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('best.pt').to(device)
```
**효과**: 추론 속도 5-10배 향상 (GPU 사용 시)

### 4. 멀티스레딩
```python
from threading import Thread

# 카메라 캡처 스레드
def capture_thread():
    while running:
        ret, frame = cap.read()
        frame_queue.put(frame)

# 추론 스레드
def inference_thread():
    while running:
        frame = frame_queue.get()
        results = model(frame)
```
**효과**: I/O 대기 시간 감소

---

## 🎓 학술적 가치

### 논문 작성 시 강조 사항

1. **실시간 성능**
   - 평균 FPS: XX FPS
   - 추론 지연 시간: XX ms
   - 실용성 입증

2. **현장 적용 가능성**
   - 저사양 노트북에서도 실시간 처리
   - 별도 서버 없이 엣지 디바이스 배포 가능
   - 실제 건설 현장 모니터링 시나리오

3. **사용자 경험**
   - 직관적인 UI/UX
   - 즉각적인 피드백 (실시간 경고)
   - 키보드 단축키로 편리한 제어

---

## 🔄 다음 단계

### 단기 (1주일)
1. ✅ 기본 웹캠 추론 구현
2. ✅ 통계 오버레이 추가
3. ✅ 키보드 컨트롤 구현
4. ✅ 스크린샷 저장 기능

### 중기 (2주일)
1. 녹화 기능 추가
2. Telegram 알림 연동
3. 성능 최적화 (GPU, 멀티스레딩)
4. 다중 카메라 지원

### 장기 (1개월)
1. 웹 대시보드 통합 (Streamlit)
2. CCTV/IP 카메라 연동
3. 탐지 히스토리 및 통계 분석
4. 배치 처리 최적화

---

## 📚 참고 자료

- [OpenCV VideoCapture 공식 문서](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [YOLOv8 실시간 추론 가이드](https://docs.ultralytics.com/modes/predict/#streaming-source)
- [프로젝트 추론 시스템](../../src/inference/inference.py)
- [Streamlit 웹 인터페이스](../../src/web_interface/app.py)

---

## 📝 변경 이력

| 날짜 | 버전 | 변경 내용 | 작성자 |
|------|------|-----------|--------|
| 2025-11-23 | 1.0 | 초안 작성 | SafetyVisionAI |

---

**작성자**: SafetyVisionAI Team
**최종 수정**: 2025-11-23
**상태**: 진행 중 (Phase 8)
