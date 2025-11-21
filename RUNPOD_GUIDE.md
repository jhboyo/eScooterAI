# RunPod 훈련 가이드

RunPod GPU 환경에서 Safety Vision AI 프로젝트를 설정하고 훈련하는 방법입니다.

## 사전 요구사항

- RunPod GPU 인스턴스 (A100)


---

## 설정 순서

### 1. Git 설치 (필요시)

```bash
# Ubuntu/Debian
apt-get update && apt-get install -y git

# 또는 conda 환경
conda install -c anaconda git
```

### 2. 프로젝트 Clone

```bash
git clone https://github.com/jhboyo/SafetyVisionAI.git
cd SafetyVisionAI
```

### 3. uv 설치

```bash
# uv 패키지 관리자 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 환경 변수 적용
source ~/.bashrc
# 또는
source ~/.zshrc
```

### 4. Python 의존성 설치

```bash
uv sync
```

### 5. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# PROJECT_ROOT를 현재 경로로 자동 설정
sed -i "s|PROJECT_ROOT=.*|PROJECT_ROOT=$(pwd)|" .env

# 확인
cat .env
```

### 6. Hugging Face 데이터셋 다운로드

```bash
# huggingface-hub CLI 설치
uv tool install huggingface-hub

# Hugging Face 로그인 (Rate Limit 방지를 위해 권장)
# 토큰 발급: https://huggingface.co/settings/tokens
uv tool run hf auth login

# 데이터셋 다운로드 (약 1.7GB)
uv tool run hf download jhboyo/ppe-dataset --repo-type dataset --local-dir ./dataset/data --max-workers 4
```

### 7. 데이터셋 YAML 생성

```bash
# ppe_dataset.yaml 생성 (절대 경로 포함)
uv run python src/1_preprocess/step5_generate_yaml.py
```

### 8. 모델 훈련 실행

```bash
uv run python src/2_training/train.py --data configs/ppe_dataset.yaml
```

---

## 훈련 설정 조정 (선택)

### A100 GPU 최적화

`configs/train_config.yaml` 파일 수정:

```yaml
train:
  epochs: 100
  batch_size: 64      # A100 80GB: 64~128 가능
  workers: 8          # CPU 코어 수에 맞게 조정
  device: 0           # GPU 사용
  imgsz: 640          # 이미지 크기
```

### 멀티 GPU 사용

```yaml
train:
  device: 0,1         # GPU 2개 사용
  # 또는
  device: 0,1,2,3     # GPU 4개 사용
```

---

## 훈련 결과 확인

### 결과 파일 위치

```bash
ls -la models/ppe_detection/
```

| 파일 | 설명 |
|------|------|
| `weights/best.pt` | 최고 성능 모델 (배포용) |
| `weights/last.pt` | 마지막 체크포인트 |
| `results.csv` | 에포크별 훈련 통계 |
| `confusion_matrix.png` | 혼동 행렬 |
| `PR_curve.png` | Precision-Recall 곡선 |

### 결과 시각화

```bash
uv run python src/2_training/visualize_results.py
```

---

## 전체 명령어 (원라인)

```bash
apt-get update && apt-get install -y git && \
git clone https://github.com/jhboyo/SafetyVisionAI.git && \
cd SafetyVisionAI && \
curl -LsSf https://astral.sh/uv/install.sh | sh && \
source ~/.bashrc && \
uv sync && \
cp .env.example .env && \
sed -i "s|PROJECT_ROOT=.*|PROJECT_ROOT=$(pwd)|" .env && \
uv tool install huggingface-hub && \
uv tool run hf download jhboyo/ppe-dataset --repo-type dataset --local-dir ./dataset/data && \
uv run python src/1_preprocess/step5_generate_yaml.py && \
uv run python src/2_training/train.py --data configs/ppe_dataset.yaml
```

---

## 문제 해결

### uv 명령어를 찾을 수 없음

```bash
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc
```

### CUDA 메모리 부족

`configs/train_config.yaml`에서 batch_size를 줄이세요:

```yaml
train:
  batch_size: 16  # 또는 8
```

### 데이터셋 경로 오류

```bash
# YAML 파일 재생성
uv run python src/1_preprocess/step5_generate_yaml.py

# 경로 확인
cat configs/ppe_dataset.yaml
```

---

## 훈련 완료 후

### Hugging Face에 모델 업로드

훈련된 모델을 Hugging Face에 업로드하여 관리할 수 있습니다. 데이터셋(`jhboyo/ppe-dataset`)과 분리된 별도의 모델 저장소를 사용하는 것을 권장합니다.

```bash
# 전체 훈련 결과 폴더 업로드 (새 저장소 자동 생성)
uv tool run hf upload jhboyo/ppe-detection-model ./models/ppe_detection \
    --repo-type model \
    --commit-message "Add trained YOLOv8 PPE detection model"

# 또는 best.pt 파일만 업로드
uv tool run hf upload jhboyo/ppe-detection-model ./models/ppe_detection/weights/best.pt \
    --repo-type model
```

### 모델 다운로드

훈련이 완료되면 `models/ppe_detection/weights/best.pt` 파일을 로컬로 다운로드하세요.

```bash
# Hugging Face에서 모델 다운로드
uv tool run hf download jhboyo/ppe-detection-model --repo-type model --local-dir ./models
```

### 추론 테스트

```bash
uv run python src/3_inference/inference.py \
    --model models/ppe_detection/weights/best.pt \
    --input dataset/data/test/images/
```

---

## 예상 훈련 시간

| GPU | Batch Size | 100 Epochs |
|-----|------------|------------|
| A100 80GB | 64 | ~1시간 |
| A100 40GB | 32 | ~3-4시간 |
| T4 16GB | 16 | ~8-10시간 |

---

## 참고

- [YOLOv8 공식 문서](https://docs.ultralytics.com/)
- [RunPod 문서](https://docs.runpod.io/)
- [Hugging Face Datasets](https://huggingface.co/datasets/jhboyo/ppe-dataset)
