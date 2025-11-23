# Hugging Face Spaces 배포 가이드

> Safety Vision AI를 Hugging Face Spaces에 무료로 배포하는 완벽 가이드

---

## 📋 목차

1. [사전 준비사항](#1-사전-준비사항)
2. [Hugging Face 계정 생성](#2-hugging-face-계정-생성)
3. [배포 파일 준비](#3-배포-파일-준비)
4. [Space 생성 및 설정](#4-space-생성-및-설정)
5. [파일 업로드](#5-파일-업로드)
6. [⭐ 업데이트 및 재배포](#6-업데이트-및-재배포-중요) **(중요!)**
7. [배포 확인 및 테스트](#7-배포-확인-및-테스트)
8. [문제 해결](#8-문제-해결)
9. [최적화 팁](#9-최적화-팁)

---

## 1. 사전 준비사항

### ✅ 필요한 것

- [ ] Hugging Face 계정 (무료)
- [ ] Git 설치
- [ ] 훈련된 YOLOv8 모델 (`models/ppe_detection/weights/best.pt`)
- [ ] 인터넷 연결

### 📂 현재 프로젝트 구조 확인

```
SafetyVisionAI/
├── src/
│   └── 5_web_interface/
│       ├── app.py                  # 메인 앱
│       ├── components/             # UI 컴포넌트
│       ├── utils/                  # 유틸리티
│       └── assets/                 # 정적 파일
├── models/
│   └── ppe_detection/
│       └── weights/
│           └── best.pt             # 훈련된 모델
├── pyproject.toml
└── .streamlit/
    └── config.toml
```

---

## 2. Hugging Face 계정 생성

### Step 1: 회원가입

1. https://huggingface.co/ 접속
2. 우측 상단 **Sign Up** 클릭
3. 이메일 또는 GitHub 계정으로 가입
4. 이메일 인증 완료

### Step 2: Access Token 생성 (선택사항)

1. 프로필 아이콘 → **Settings** 클릭
2. 좌측 메뉴 **Access Tokens** 선택
3. **New token** 클릭
   - Name: `safety-vision-ai-deploy`
   - Role: `write` 선택
4. 토큰 복사 및 안전하게 저장 (한 번만 표시됨)

---

## 3. 배포 파일 준비

### Step 1: requirements.txt 생성

프로젝트 루트에서 실행:

```bash
# 현재 디렉토리 확인
pwd
# /Users/joonho/workspace/sogang/tf-basic/SafetyVisionAI

# requirements.txt 생성
uv pip compile pyproject.toml -o requirements.txt
```

**또는 수동으로 작성:**

```bash
cat > requirements.txt << 'EOF'
ultralytics==8.3.229
streamlit==1.51.0
plotly==6.5.0
pillow>=9.5.0
opencv-python>=4.8.0
numpy>=1.24.0
python-dotenv>=1.2.1
pandas>=2.1.0
EOF
```

**⚠️ 중요:**
- `tensorflow` 제거 (Streamlit 앱에서 미사용)
- `huggingface-hub`, `hf-transfer` 제거 (불필요)
- 경량화된 의존성만 포함

### Step 2: README.md 작성 (Space 설명용)

```bash
cat > SPACE_README.md << 'EOF'
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

# 🏗️ SafetyVisionAI - PPE Detection System

딥러닝 기반 건설현장 안전 장비 착용 모니터링 시스템

## 🎯 기능

- ⛑️ 헬멧 착용 감지
- 🦺 안전조끼 착용 감지
- 📊 실시간 안전 수준 평가
- 📈 통계 및 시각화

## 🚀 사용 방법

1. 좌측 사이드바에서 모델과 설정 선택
2. 이미지 업로드 (JPG, PNG 지원)
3. "🚀 탐지 시작" 버튼 클릭
4. 결과 확인

## 📚 기술 스택

- YOLOv8 (Ultralytics)
- Streamlit
- OpenCV
- Plotly

## 👥 팀

TensorGuard


EOF
```

### Step 3: 배포용 디렉토리 구조 생성

프로젝트 루트에 `deploy/` 디렉토리 생성:

```bash
# 배포 디렉토리 생성
mkdir -p deploy/huggingface

# 필요한 파일 복사
cp -r src/5_web_interface/* deploy/huggingface/
cp requirements.txt deploy/huggingface/
cp SPACE_README.md deploy/huggingface/README.md

# 모델 파일 복사
mkdir -p deploy/huggingface/models/ppe_detection/weights
cp models/ppe_detection/weights/best.pt deploy/huggingface/models/ppe_detection/weights/

# .streamlit 설정 복사
cp -r .streamlit deploy/huggingface/
```

**최종 구조:**

```
deploy/huggingface/
├── app.py                          # 메인 앱
├── README.md                       # Space 설명 (SPACE_README.md)
├── requirements.txt                # 의존성
├── components/
│   ├── __init__.py
│   ├── uploader.py
│   └── statistics.py
├── utils/
│   ├── __init__.py
│   ├── inference.py
│   └── plotting.py
├── assets/
│   └── styles.css
├── models/
│   └── ppe_detection/
│       └── weights/
│           └── best.pt             # 훈련된 모델
└── .streamlit/
    └── config.toml
```

### Step 4: app.py 경로 수정 (필요시)

`deploy/huggingface/app.py` 파일을 열고, 상대 경로가 올바른지 확인:

```python
# 모델 경로 확인
MODEL_DIR = Path(__file__).parent / "models" / "ppe_detection" / "weights"

# CSS 파일 경로 확인
CSS_FILE = Path(__file__).parent / "assets" / "styles.css"
```

**✅ 이미 올바르게 설정되어 있으면 수정 불필요**

---

## 4. Space 생성 및 설정

### Step 1: Space 생성

1. https://huggingface.co/spaces 접속
2. **Create new Space** 클릭
3. 정보 입력:
   - **Owner**: 본인 계정
   - **Space name**: `safety-vision-ai` (또는 원하는 이름)
   - **License**: `mit` (오픈소스)
   - **Select the Space SDK**: `Streamlit` 선택
   - **Space hardware**: `CPU basic - Free` 선택
   - **Repo type**: `Public` (무료)
4. **Create Space** 클릭

### Step 2: Git 저장소 클론

터미널에서 실행:

```bash
# Space Git 저장소 클론
git clone https://huggingface.co/spaces/YOUR_USERNAME/safety-vision-ai
cd safety-vision-ai

# 예시
# git clone https://huggingface.co/spaces/jhboyo/safety-vision-ai
# cd safety-vision-ai
```

---

## 5. 파일 업로드

### 방법 1: Git 사용 (추천)

```bash
# 1. 클론한 Space 디렉토리로 이동
cd safety-vision-ai

# 2. 배포 파일 복사
cp -r ../SafetyVisionAI/deploy/huggingface/* .

# 3. Git LFS 설정 (대용량 모델 파일용)
git lfs install
git lfs track "*.pt"
git add .gitattributes

# 4. 모든 파일 추가
git add .

# 5. 커밋
git commit -m "Initial deployment: Safety Vision AI with YOLOv8"

# 6. Hugging Face에 푸시
git push

# 인증 요구 시:
# Username: YOUR_HUGGINGFACE_USERNAME
# Password: YOUR_ACCESS_TOKEN (Step 2에서 생성한 토큰)
```

### 방법 2: Web UI 사용 (간단하지만 느림)

1. Space 페이지에서 **Files** 탭 클릭
2. **Add file** → **Upload files** 선택
3. `deploy/huggingface/` 내 모든 파일을 드래그 앤 드롭
4. Commit message 입력: `Initial deployment`
5. **Commit changes to main** 클릭

**⚠️ 주의:** 대용량 파일 (best.pt)은 Git LFS 필요하므로 방법 1 권장

---

## 6. 업데이트 및 재배포 (중요!)

로컬에서 코드를 수정한 후 Hugging Face Spaces에 반영하는 절차입니다.

### 📝 업데이트 워크플로우

```
로컬 수정 → deploy/ 동기화 → Git 푸시 → 자동 재배포
```

### Step 1: 로컬에서 코드 수정

```bash
# 로컬 프로젝트에서 작업
cd /Users/joonho/workspace/sogang/tf-basic/SafetyVisionAI

# 예시: 웹 인터페이스 수정
vim src/5_web_interface/app.py
vim src/5_web_interface/utils/inference.py
# 또는 VS Code 등 에디터 사용
```

### Step 2: 수정사항을 deploy/ 디렉토리에 동기화

```bash
# 프로젝트 루트에서 실행
cd /Users/joonho/workspace/sogang/tf-basic/SafetyVisionAI

# 방법 1: 전체 동기화 (권장)
# src/5_web_interface의 모든 파일을 deploy/huggingface로 복사
rsync -av --delete src/5_web_interface/ deploy/huggingface/

# 방법 2: 특정 파일만 복사
cp src/5_web_interface/app.py deploy/huggingface/
cp src/5_web_interface/utils/inference.py deploy/huggingface/utils/

# 모델 파일이 변경된 경우
cp models/ppe_detection/weights/best.pt deploy/huggingface/models/ppe_detection/weights/
```

**⚠️ 주의사항:**
- `rsync -av --delete`는 deploy/huggingface를 src/5_web_interface와 완전히 동기화
- `--delete` 옵션은 로컬에 없는 파일을 deploy에서도 삭제
- requirements.txt나 README.md는 별도 관리되므로 주의

### Step 3: safety-vision-ai Git 저장소에 푸시

```bash
# Hugging Face Space Git 저장소로 이동
cd /Users/joonho/workspace/sogang/tf-basic/SafetyVisionAI/safety-vision-ai

# deploy/huggingface의 변경사항을 여기로 복사
rsync -av --exclude='.git' ../deploy/huggingface/ .

# 변경사항 확인
git status
git diff

# 변경된 파일 추가
git add .

# 커밋 메시지 작성 (의미 있게)
git commit -m "Update: 신뢰도 임계값 UI 개선"
# 또는
git commit -m "Fix: 모델 로딩 에러 수정"
# 또는
git commit -m "Feature: 디버그 모드 추가"

# Hugging Face에 푸시
git push

# 인증 정보 입력 (처음 1회)
# Username: YOUR_HUGGINGFACE_USERNAME
# Password: YOUR_ACCESS_TOKEN
```

### Step 4: 자동 재배포 확인

1. **Hugging Face Space 페이지 접속**
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/safety-vision-ai
   ```

2. **Logs 탭 확인**
   - 푸시 직후 자동으로 재빌드 시작
   - 빌드 진행 상황 실시간 확인:
     ```
     Updating repository...
     Installing dependencies...
     Restarting application...
     ```

3. **빌드 완료 대기**
   - 예상 시간: 30초~2분 (변경 범위에 따라 다름)
   - **✅ Running** 상태로 변경되면 완료

4. **변경사항 테스트**
   - Space URL 새로고침 (Ctrl+Shift+R로 캐시 무시)
   - 수정한 기능 정상 작동 확인
   - 에러 로그 확인

---

### 🚀 빠른 업데이트 스크립트 (권장)

반복 작업을 자동화하는 스크립트를 만들어 사용하세요.

**`scripts/deploy_to_hf.sh` 생성:**

```bash
#!/bin/bash
# Hugging Face Spaces 배포 자동화 스크립트

set -e  # 에러 발생 시 중단

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 프로젝트 루트 경로
PROJECT_ROOT="/Users/joonho/workspace/sogang/tf-basic/SafetyVisionAI"
DEPLOY_DIR="$PROJECT_ROOT/deploy/huggingface"
HF_REPO="$PROJECT_ROOT/safety-vision-ai"

echo -e "${GREEN}=== Hugging Face Spaces 배포 시작 ===${NC}"

# Step 1: 로컬 → deploy 동기화
echo -e "${YELLOW}Step 1: 로컬 변경사항을 deploy 디렉토리로 동기화...${NC}"
rsync -av --delete \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    "$PROJECT_ROOT/src/5_web_interface/" "$DEPLOY_DIR/"

echo -e "${GREEN}✓ 동기화 완료${NC}"

# Step 2: deploy → Hugging Face Git 저장소 동기화
echo -e "${YELLOW}Step 2: deploy 디렉토리를 Hugging Face 저장소로 복사...${NC}"
cd "$HF_REPO"
rsync -av --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    "$DEPLOY_DIR/" .

echo -e "${GREEN}✓ 복사 완료${NC}"

# Step 3: Git 변경사항 확인
echo -e "${YELLOW}Step 3: Git 변경사항 확인...${NC}"
git status

# Step 4: 커밋 메시지 입력받기
echo -e "${YELLOW}Step 4: 커밋 메시지를 입력하세요:${NC}"
read -p "Commit message: " COMMIT_MSG

if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Update: $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${YELLOW}기본 메시지 사용: $COMMIT_MSG${NC}"
fi

# Step 5: Git 커밋 및 푸시
echo -e "${YELLOW}Step 5: Git 커밋 및 푸시...${NC}"
git add .
git commit -m "$COMMIT_MSG" || echo "변경사항 없음 또는 커밋 실패"
git push

echo -e "${GREEN}=== 배포 완료! ===${NC}"
echo -e "${GREEN}Hugging Face Spaces에서 자동으로 재빌드됩니다.${NC}"
echo -e "${GREEN}확인: https://huggingface.co/spaces/jhboyo/safety-vision-ai${NC}"
```

**실행 권한 부여:**

```bash
chmod +x scripts/deploy_to_hf.sh
```

**사용법:**

```bash
# 로컬에서 코드 수정 후
cd /Users/joonho/workspace/sogang/tf-basic/SafetyVisionAI
./scripts/deploy_to_hf.sh

# 커밋 메시지 입력 프롬프트에서:
# Commit message: UI 개선 및 버그 수정

# 자동으로 동기화 → 커밋 → 푸시 완료!
```

---

### 🔄 업데이트 시나리오별 가이드

#### 시나리오 1: UI 코드만 수정 (빠름)

```bash
# app.py 수정
vim src/5_web_interface/app.py

# 배포 스크립트 실행
./scripts/deploy_to_hf.sh
# Commit message: UI 레이아웃 개선

# 예상 재빌드 시간: 30초~1분
```

#### 시나리오 2: 의존성 추가 (중간)

```bash
# pyproject.toml 수정
uv add new-package

# requirements.txt 재생성
uv pip compile pyproject.toml -o requirements.txt

# deploy로 복사
cp requirements.txt deploy/huggingface/

# 배포 스크립트 실행
./scripts/deploy_to_hf.sh
# Commit message: Add new-package dependency

# 예상 재빌드 시간: 1~2분
```

#### 시나리오 3: 모델 파일 변경 (특별한 절차 필요!)

**⚠️ 중요: 모델 파일(best.pt)은 크기가 크므로 일반 Git으로 푸시 불가!**

**방법 1: Web UI 업로드 (권장, 가장 간단)**

```bash
# 1. 로컬에서 모델 재훈련 완료
# 2. Web UI로 수동 업로드

1. https://huggingface.co/spaces/jhboyo/safety-vision-ai/tree/main
2. "Files" 탭 클릭
3. models/ppe_detection/weights/ 경로로 이동
4. "Upload files" 버튼 클릭
5. best.pt 파일 드래그 앤 드롭
6. Commit message: "Update model to v2.0"
7. "Commit changes" 클릭

# 예상 재빌드 시간: 2~5분
```

**방법 2: Git LFS 사용 (고급 사용자)**

```bash
# Git LFS 설정 (최초 1회)
cd safety-vision-ai
git lfs install
git lfs track "*.pt"
git add .gitattributes

# 모델 파일 복사 및 푸시
cp ../models/ppe_detection/weights/best.pt models/ppe_detection/weights/
git add models/ppe_detection/weights/best.pt
git commit -m "Update model to v2.0"
git push

# ⚠️ 주의: 무료 계정은 LFS 스토리지 1GB 제한
```

**방법 3: 배포 스크립트 사용 (대화형)**

```bash
./scripts/deploy_to_hf.sh

# 프롬프트에서:
모델 파일(best.pt)을 업데이트하시겠습니까? (y/N): y
선택하세요:
  1) Web UI로 수동 업로드 (권장)
  2) Git LFS로 푸시 (고급)
  3) 건너뛰기
선택 (1-3): 1

# Web UI 링크 표시, 업로드 후 Enter
```

#### 시나리오 4: 긴급 버그 수정 (최소한의 변경)

```bash
# 특정 파일만 수정
vim src/5_web_interface/utils/inference.py

# 해당 파일만 복사
cp src/5_web_interface/utils/inference.py deploy/huggingface/utils/

cd safety-vision-ai
rsync -av ../deploy/huggingface/ .
git add utils/inference.py
git commit -m "Hotfix: Fix model loading error"
git push

# 예상 재빌드 시간: 30초~1분
```

---

### ⚠️ 주의사항

1. **로컬 프로젝트와 deploy 동기화 필수**
   - 로컬에서 수정 → 반드시 deploy로 복사
   - deploy를 건너뛰고 직접 safety-vision-ai에서 수정하면 나중에 충돌 발생

2. **모델 파일은 별도 관리 (중요!)**
   - `best.pt`는 보통 150MB~600MB로 매우 큼
   - **일반 Git으로 푸시 불가** (100MB 제한)
   - **Web UI 업로드 권장** (가장 간단하고 안전)
   - Git LFS 사용 시 무료 계정은 1GB 제한
   - `deploy_to_hf.sh` 스크립트는 모델 파일 자동 제외

3. **Git LFS 대용량 파일 관리 (선택사항)**
   - 모델 파일(*.pt)은 Git LFS로 추적 가능
   - 100MB 이상 파일은 일반 Git으로 푸시 불가
   - Web UI가 더 간단하므로 LFS는 선택사항

4. **캐시 주의**
   - 브라우저 캐시 때문에 변경사항이 안 보일 수 있음
   - **Ctrl+Shift+R** (하드 리프레시) 사용

5. **빌드 실패 시**
   - Logs 탭에서 에러 확인
   - requirements.txt 의존성 문제 확인
   - 경로 문제 확인 (상대경로 사용)

6. **롤백 방법**
   ```bash
   cd safety-vision-ai
   git log  # 이전 커밋 해시 확인
   git revert <commit-hash>  # 특정 커밋 되돌리기
   git push
   ```

---

### 📊 업데이트 체크리스트

배포 전 확인:

- [ ] 로컬에서 테스트 완료 (`uv run streamlit run src/5_web_interface/app.py`)
- [ ] 변경사항을 deploy/huggingface로 동기화
- [ ] requirements.txt 최신화 (의존성 변경 시)
- [ ] **모델 파일 변경 시: Web UI로 별도 업로드 계획 확인**
- [ ] 커밋 메시지 명확하게 작성
- [ ] .gitignore 확인 (불필요한 파일 제외)

배포 후 확인:

- [ ] Logs에서 빌드 성공 확인
- [ ] Space URL에서 변경사항 확인
- [ ] 하드 리프레시 (Ctrl+Shift+R)
- [ ] 주요 기능 테스트
- [ ] 에러 로그 없는지 확인

---

## 7. 배포 확인 및 테스트

### Step 1: 빌드 로그 확인

1. Space 페이지에서 **Logs** 탭 클릭
2. 빌드 진행 상황 확인:
   ```
   Building image...
   Installing dependencies...
   Starting Streamlit...
   ```
3. **✅ Running** 상태가 되면 성공

**예상 빌드 시간:** 3~5분

### Step 2: 앱 접속

1. Space URL 접속:
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/safety-vision-ai
   ```
2. Streamlit UI 로드 확인
3. 사이드바 설정 확인
4. 테스트 이미지 업로드
5. 탐지 실행 및 결과 확인

### Step 3: 성능 테스트

**무료 CPU 성능:**
- ✅ 모델 로드: 3~5초
- ✅ 단일 이미지 추론: 1~2초
- ✅ 배치 처리 (10장): 15~25초

**정상 작동 확인:**
- [ ] 이미지 업로드 정상
- [ ] 모델 로드 성공
- [ ] 추론 결과 표시
- [ ] 바운딩 박스 정상 표시
- [ ] 통계 정확
- [ ] CSS 스타일 적용

---

## 8. 문제 해결

### 🔴 문제 1: "No module named 'ultralytics'"

**원인:** `requirements.txt`가 누락되거나 잘못됨

**해결:**
```bash
# requirements.txt 확인
cat requirements.txt

# ultralytics 버전 확인
# ultralytics==8.3.229 이 있어야 함

# 파일 재업로드
git add requirements.txt
git commit -m "Fix requirements.txt"
git push
```

---

### 🔴 문제 2: "Model file not found"

**원인:** 모델 파일 경로 문제 또는 Git LFS 미설정

**해결:**
```bash
# 1. Git LFS 설정 확인
git lfs track "*.pt"
git add .gitattributes

# 2. 모델 파일 다시 추가
git add models/ppe_detection/weights/best.pt
git commit -m "Add model file with Git LFS"
git push

# 3. 모델 파일 크기 확인 (100KB 이상이어야 함)
ls -lh models/ppe_detection/weights/best.pt
```

**또는 파일이 너무 크면 (>5GB):**
```bash
# Hugging Face Hub에 모델만 별도 업로드 후 app.py에서 다운로드
# 예시: huggingface-cli upload
```

---

### 🔴 문제 3: "Application Error" 또는 앱 시작 실패

**원인:** `app.py` 경로 또는 설정 문제

**해결:**
```bash
# 1. README.md의 app_file 확인
cat README.md | grep app_file
# app_file: app.py 이어야 함 (src/5_web_interface/app.py 아님)

# 2. 파일 구조 확인
ls -la
# app.py가 루트에 있어야 함

# 3. Streamlit 설정 확인
cat .streamlit/config.toml
```

---

### 🔴 문제 4: CSS 파일 로드 실패

**원인:** 경로 문제

**해결:**
```python
# app.py에서 경로 수정
from pathlib import Path

CSS_FILE = Path(__file__).parent / "assets" / "styles.css"

# 디버그
print(f"CSS path: {CSS_FILE}")
print(f"CSS exists: {CSS_FILE.exists()}")
```

---

### 🔴 문제 5: "Out of Memory" 오류

**원인:** 무료 CPU는 16GB RAM이지만, 여러 사용자 동시 접속 시 부족 가능

**해결:**
```python
# app.py에서 메모리 최적화
import gc
import torch

# 추론 후 메모리 해제
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 배치 크기 줄이기
# batch_size = 4 → batch_size = 1
```

---

### 🔴 문제 6: "Space is sleeping"

**원인:** 48시간 미사용 시 자동 sleep

**해결:**
- 누군가 접속하면 자동으로 깨어남 (10~30초 소요)
- 또는 GitHub Actions로 주기적 Ping:

```yaml
# .github/workflows/keep-alive.yml
name: Keep Hugging Face Space Alive

on:
  schedule:
    - cron: '0 */12 * * *'  # 12시간마다 실행

jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - name: Ping Space
        run: |
          curl https://huggingface.co/spaces/YOUR_USERNAME/safety-vision-ai
```

---

## 9. 최적화 팁

### 🚀 성능 최적화

#### 1. 모델 경량화

```bash
# YOLOv8n (Nano) 사용 - 가장 빠름
# best.pt가 YOLOv8n 기반이면 OK

# 또는 모델 양자화 (선택사항)
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx", dynamic=True)  # ONNX로 변환 (더 빠름)
```

#### 2. 이미지 전처리 최적화

```python
# app.py에서 이미지 크기 제한
MAX_IMAGE_SIZE = (1280, 1280)  # YOLOv8 기본 크기

def preprocess_image(image):
    if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
        image.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
    return image
```

#### 3. 캐싱 강화

```python
# utils/inference.py에서
@st.cache_resource(ttl=3600)  # 1시간 캐시
def load_model(model_path: str):
    return YOLO(model_path)

@st.cache_data(ttl=600)  # 10분 캐시
def run_inference_cached(image_hash, model_path, conf):
    # 동일 이미지 재추론 방지
    pass
```

---

### 🎨 UI/UX 개선

#### 1. 로딩 애니메이션 추가

```python
with st.spinner("🔄 모델 로딩 중..."):
    model = load_model(model_path)

with st.spinner("🎯 이미지 분석 중..."):
    results = run_inference(image, model)
```

#### 2. 에러 메시지 개선

```python
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"❌ 모델 로드 실패: {str(e)}")
    st.info("💡 관리자에게 문의하세요: example@email.com")
    st.stop()
```

#### 3. 다국어 지원 (선택사항)

```python
# config.py
LANGUAGES = {
    "ko": {
        "title": "안전 비전 AI",
        "upload": "이미지 업로드",
    },
    "en": {
        "title": "Safety Vision AI",
        "upload": "Upload Images",
    }
}
```

---

### 📊 분석 및 모니터링

#### 1. Hugging Face Analytics 활성화

Space 설정에서 Analytics 활성화:
- 방문자 수
- 사용 시간
- 인기 시간대

#### 2. 사용자 피드백 수집

```python
# app.py에 피드백 섹션 추가
with st.sidebar:
    st.markdown("---")
    st.subheader("📝 피드백")
    feedback = st.text_area("의견을 남겨주세요")
    if st.button("제출"):
        # Google Forms 또는 이메일로 전송
        st.success("감사합니다!")
```

---

### 🔒 보안 및 안정성

#### 1. Rate Limiting (선택사항)

```python
# utils/rate_limit.py
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests=10, window=60):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)

    def is_allowed(self, user_id):
        now = time.time()
        # 오래된 요청 제거
        self.requests[user_id] = [
            t for t in self.requests[user_id]
            if now - t < self.window
        ]

        if len(self.requests[user_id]) < self.max_requests:
            self.requests[user_id].append(now)
            return True
        return False
```

#### 2. 입력 검증 강화

```python
# components/uploader.py
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"]

def validate_image(file):
    # 크기 검증
    if file.size > MAX_FILE_SIZE:
        raise ValueError(f"파일이 너무 큽니다 (최대 {MAX_FILE_SIZE/1024/1024}MB)")

    # 확장자 검증
    ext = file.name.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"지원하지 않는 형식입니다: {ext}")

    # 실제 이미지 파일인지 검증
    try:
        img = Image.open(file)
        img.verify()
    except:
        raise ValueError("손상된 이미지 파일입니다")
```

---

## 10. 커스텀 도메인 연결 (선택사항)

### Cloudflare + Hugging Face 조합

1. **Cloudflare 계정 생성** (무료)
2. **도메인 구매** (선택) 또는 무료 도메인 (Freenom 등)
3. **DNS 설정:**
   ```
   Type: CNAME
   Name: ai
   Target: huggingface.co
   Proxied: Yes (오렌지 구름)
   ```
4. **Hugging Face Space 설정:**
   - Space Settings → Custom Domain
   - `ai.yourdomain.com` 입력
   - DNS 확인 후 활성화

**최종 URL:** `https://ai.yourdomain.com`

---

## 11. 고급: GPU 무료 신청 (선택사항)

### Community GPU Grant 신청

1. Space 페이지 → **Settings** 탭
2. **Request GPU** 클릭
3. 신청서 작성:
   - **Project Description**: 건설현장 안전 모니터링 학술 프로젝트
   - **Why GPU**: 실시간 추론 속도 향상 필요
   - **Public Benefit**: 오픈소스 교육 자료로 공개
4. 제출 후 1~2주 내 승인 여부 통보

**승인되면:**
- T4 GPU (16GB VRAM) 무료 사용
- 추론 속도 10배 향상 (0.1~0.2초/이미지)

---

## 12. 최종 체크리스트

배포 전 확인:

- [ ] Hugging Face 계정 생성 완료
- [ ] requirements.txt 생성 완료
- [ ] SPACE_README.md 작성 완료
- [ ] 배포 디렉토리 구조 준비 완료
- [ ] 모델 파일 (best.pt) 포함
- [ ] Git LFS 설정 완료
- [ ] Space 생성 완료
- [ ] 파일 업로드 완료
- [ ] 빌드 성공 확인
- [ ] 앱 정상 작동 확인
- [ ] 테스트 이미지 추론 성공

배포 후 확인:

- [ ] URL 공유 가능 확인
- [ ] 성능 테스트 완료
- [ ] 에러 로그 확인
- [ ] 사용자 피드백 수집
- [ ] README 업데이트 (Space URL 추가)

---

## 13. 참고 자료

### 공식 문서

- [Hugging Face Spaces 문서](https://huggingface.co/docs/hub/spaces)
- [Streamlit on Spaces 가이드](https://huggingface.co/docs/hub/spaces-sdks-streamlit)
- [Git LFS 문서](https://git-lfs.github.com/)
- [YOLOv8 문서](https://docs.ultralytics.com/)

### 예시 Spaces

- [Object Detection Demo](https://huggingface.co/spaces/Gradio-Blocks/Object-Detection-With-YOLOV8)
- [Image Classification](https://huggingface.co/spaces/streamlit/image-classification)

### 커뮤니티

- [Hugging Face Discord](https://discord.com/invite/hugging-face)
- [Hugging Face Forums](https://discuss.huggingface.co/)

---

## 📞 지원

**문제 발생 시:**

1. **Logs 확인**: Space 페이지 → Logs 탭
2. **Community 검색**: https://discuss.huggingface.co/
3. **Issue 등록**: Space 페이지 → Community 탭

---

## ✅ 완료!

축하합니다! 🎉

이제 Safety Vision AI가 Hugging Face Spaces에서 실행 중입니다.

**최종 URL:**
```
https://huggingface.co/spaces/YOUR_USERNAME/safety-vision-ai
```

**공유하기:**
- 프로젝트 README.md에 URL 추가
- 논문에 데모 링크 삽입
- 팀원들과 공유

**다음 단계:**
- [ ] 성능 모니터링
- [ ] 사용자 피드백 수집
- [ ] UI/UX 개선
- [ ] GPU Grant 신청 (선택)
- [ ] 커스텀 도메인 연결 (선택)

---

**Last Updated**: 2025-11-23
**Version**: 1.0
**Author**: SafetyVisionAI Team
**License**: MIT
