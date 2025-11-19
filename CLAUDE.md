# Safety Vision AI - 딥러닝 기반 건설현장 안전 장비 착용 모니터링 플랫폼

## 프로젝트 개요
딥러닝 기반의 작업자 개인보호구(PPE) 착용 감지 및 산업안전 재해 방지를 위한 머신러닝 모델 개발 프로젝트
README.md 파일을 자세히 참고하여 프로젝트 진행 내용을 참고하여 답변해야 함

## 주요 기능
- 개인보호구(헬멧, 안전조끼) 착용 상태 감지
- 실시간 안전 모니터링

## 탐지 대상 클래스
```yaml
classes:
  0: helmet      # 헬멧 착용
  1: vest        # 안전조끼 착용
```

## 데이터셋
- Hard Hat Detection (Kaggle): helmet, head 클래스
- Safety Helmet and Reflective Jacket (Kaggle): helmet, vest 클래스

## 프로젝트 구조
```
SafetyVisionAI/
├── materials/          # 프로젝트 관련 문서 및 자료
├── images/             # 데이터셋
│   ├── raw/           # 원본 데이터 (Kaggle 다운로드)
│   ├── processed/     # 전처리된 데이터
│   ├── train/         # 훈련 데이터
│   │   ├── images/
│   │   └── labels/
│   ├── val/           # 검증 데이터
│   │   ├── images/
│   │   └── labels/
│   └── test/          # 테스트 데이터
│       ├── images/
│       └── labels/
├── models/            # 훈련된 모델 파일
├── src/               # 소스 코드
├── notebooks/         # Jupyter 노트북 (실험, 데이터 분석)
├── configs/           # 설정 파일
├── pyproject.toml     # Python 의존성 및 프로젝트 설정 (uv 사용)
├── uv.lock           # 의존성 락파일
├── main.py           # 메인 실행 파일
├── CLAUDE.md         # 프로젝트 지침서
└── README.md         # 프로젝트 설명
```

## 개발 환경
- Python 3.8+
- uv (패키지 관리자)
- TensorFlow/PyTorch
- OpenCV
- YOLO/Fast R-CNN 등 객체 탐지 모델

## 환경 설정
```bash
# uv로 의존성 설치
uv sync

# 가상환경 활성화 (자동으로 관리됨)
uv run python main.py
```

## 명령어
### 모델 훈련
```bash
uv run python src/train.py --config configs/train_config.yaml
```

### 추론/테스트
```bash
uv run python src/inference.py --model models/best_model.pth --input data/test_images/
```

### 데이터 전처리
```bash
uv run python src/preprocess.py --input data/raw/ --output data/processed/
```

### 메인 애플리케이션 실행
```bash
uv run python main.py
```

## 참고 자료
- materials/3조_팀소개_팀플주제_선정.pdf
