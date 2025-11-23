---
title: Safety Vision AI
emoji: 🏗️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.51.0"
app_file: app.py
pinned: false
---

# 🏗️ Safety Vision AI - PPE Detection System

딥러닝 기반 건설현장 안전 장비 착용 모니터링 시스템

## 🎯 기능

- ⛑️ **헬멧 착용 감지** - YOLOv8 기반 실시간 헬멧 탐지
- 🦺 **안전조끼 착용 감지** - 작업자 안전조끼 착용 여부 확인
- 📊 **실시간 안전 수준 평가** - 헬멧 착용률 기반 안전 등급 (Excellent/Caution/Dangerous)
- 📈 **통계 및 시각화** - 탐지 결과 통계 및 비교 뷰 제공

## 🚀 사용 방법

### 1단계: 설정

좌측 사이드바에서 다음을 설정하세요:
- **모델 선택**: `best.pt` (기본값) 또는 `last.pt`
- **신뢰도 임계값**: 0.1 ~ 1.0 (기본값: 0.25)
- **고급 옵션**: IoU 임계값, 최대 탐지 개수 (선택사항)

### 2단계: 이미지 업로드

- 📁 드래그 앤 드롭으로 이미지 업로드
- 지원 형식: JPG, JPEG, PNG, WEBP, BMP
- 최대 파일 크기: 10MB per file
- 다중 파일 업로드 지원

### 3단계: 탐지 시작

- 🚀 "탐지 시작" 버튼 클릭
- 실시간 진행 상태 확인
- 추론 결과 및 통계 확인

### 4단계: 결과 확인

- 원본 이미지와 탐지 결과 비교
- 바운딩 박스 및 클래스 확인
- 헬멧 착용률 및 안전 수준 평가

## 📚 기술 스택

| 구분 | 기술 |
|------|------|
| **딥러닝 모델** | YOLOv8 (Ultralytics) |
| **웹 프레임워크** | Streamlit |
| **이미지 처리** | OpenCV, Pillow |
| **데이터 시각화** | Plotly |
| **언어** | Python 3.11+ |

## 🎓 프로젝트 정보

### 팀

- **대학**: 서강대학교
- **프로젝트**: 딥러닝 기반 산업안전 모니터링
- **팀**: 3조

### 탐지 클래스

```yaml
classes:
  0: helmet      # 헬멧 착용
  1: vest        # 안전조끼 착용
```

### 데이터셋

- Hard Hat Detection (Kaggle)
- Safety Helmet and Reflective Jacket (Kaggle)

## 📊 모델 성능

- **모델**: YOLOv8
- **훈련 데이터**: 건설현장 PPE 이미지
- **탐지 속도**: ~1-2초/이미지 (CPU)
- **정확도**: 실시간 안전 모니터링 가능

## 🔧 로컬 실행

```bash
# 저장소 클론
git clone https://huggingface.co/spaces/YOUR_USERNAME/safety-vision-ai
cd safety-vision-ai

# 의존성 설치
pip install -r requirements.txt

# 앱 실행
streamlit run app.py
```

## 📝 라이선스

MIT License

## 📧 문의

프로젝트 관련 문의나 피드백은 이슈를 통해 남겨주세요.

---

**Made with ❤️ by SafetyVisionAI Team**
