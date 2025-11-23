# Hugging Face Spaces 배포 가이드

> Safety Vision AI - Hugging Face Spaces 배포 및 업데이트 가이드

**배포 완료 URL**: https://huggingface.co/spaces/jhboyo/safey-vision-ai

---

## 📋 목차

1. [배포 현황](#1-배포-현황)
2. [빠른 시작 - 업데이트 방법](#2-빠른-시작---업데이트-방법)
3. [상세 업데이트 가이드](#3-상세-업데이트-가이드)
4. [문제 해결](#4-문제-해결)
5. [참고 자료](#5-참고-자료)

---

## 1. 배포 현황

### ✅ 배포 완료 정보

| 항목 | 내용 |
|------|------|
| **Space URL** | https://huggingface.co/spaces/jhboyo/safey-vision-ai |
| **배포 일자** | 2025-11-23 |
| **SDK** | Streamlit 1.51.0 |
| **Hardware** | CPU basic (Free) - 16GB RAM, 2 vCPU |
| **모델** | YOLOv8n (best.pt, 6.0MB) |
| **상태** | ✅ Running (24/7 무료 호스팅) |

### 📂 배포 구조

```
SafetyVisionAI/
├── deploy/huggingface/       # 배포용 파일 (로컬)
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   ├── components/
│   ├── utils/
│   ├── assets/
│   ├── models/
│   │   └── ppe_detection/
│   │       └── weights/
│   │           └── best.pt
│   └── .streamlit/
├── safety-vision-ai/          # Hugging Face Space Git 저장소 (로컬)
│   └── (deploy/huggingface와 동일한 구조)
└── scripts/
    └── deploy_to_hf.sh        # 자동 배포 스크립트
```

---

## 2. 빠른 시작 - 업데이트 방법

### 🚀 자동 배포 스크립트 사용 (권장)

가장 빠르고 간단한 방법입니다.

```bash
# 1. 로컬에서 코드 수정
cd /Users/joonho/workspace/sogang/tf-basic/SafetyVisionAI
vim src/5_web_interface/app.py

# 2. 배포 스크립트 실행
./scripts/deploy_to_hf.sh

# 3. 커밋 메시지 입력
# Commit message: UI 개선 및 버그 수정

# 완료! Hugging Face에서 자동으로 재빌드됩니다.
```

**예상 소요 시간:**
- 스크립트 실행: ~10초
- Hugging Face 재빌드: 30초~2분

---

## 3. 상세 업데이트 가이드

### 📝 업데이트 워크플로우

```
로컬 수정 → deploy/ 동기화 → Git 푸시 → 자동 재배포
```

### Step 1: 로컬에서 코드 수정

```bash
cd /Users/joonho/workspace/sogang/tf-basic/SafetyVisionAI

# 웹 인터페이스 수정
vim src/5_web_interface/app.py
vim src/5_web_interface/utils/inference.py

# 또는 VS Code 사용
code src/5_web_interface/
```

### Step 2: 배포 스크립트 실행

```bash
./scripts/deploy_to_hf.sh
```

스크립트가 자동으로:
1. `src/5_web_interface/` → `deploy/huggingface/` 동기화
2. `deploy/huggingface/` → `safety-vision-ai/` 동기화
3. Git 변경사항 확인
4. 커밋 메시지 입력 받기
5. Git 푸시

### Step 3: 배포 확인

1. **Hugging Face Space 페이지 접속**
   ```
   https://huggingface.co/spaces/jhboyo/safey-vision-ai
   ```

2. **Logs 탭 확인**
   - 푸시 직후 자동으로 재빌드 시작
   - 빌드 진행 상황 실시간 확인

3. **빌드 완료 대기**
   - 예상 시간: 30초~2분
   - **✅ Running** 상태로 변경되면 완료

4. **변경사항 테스트**
   - Space URL 새로고침 (Ctrl+Shift+R로 캐시 무시)
   - 수정한 기능 정상 작동 확인

---

### 🔄 시나리오별 업데이트 가이드

#### 시나리오 1: UI 코드만 수정 (빠름)

```bash
# app.py 수정
vim src/5_web_interface/app.py

# 배포
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

# 배포
./scripts/deploy_to_hf.sh
# Commit message: Add new-package dependency

# 예상 재빌드 시간: 1~2분
```

#### 시나리오 3: 모델 파일 변경 ⚠️

**⚠️ 중요: 모델 파일(best.pt)은 Git LFS 또는 Web UI로 업로드 필요**

**방법 1: Web UI 업로드 (권장)**

```bash
1. https://huggingface.co/spaces/jhboyo/safey-vision-ai/tree/main 접속
2. "Files" 탭 클릭
3. models/ppe_detection/weights/ 경로로 이동
4. "Upload files" 버튼 클릭
5. best.pt 파일 드래그 앤 드롭
6. Commit message: "Update model to v2.0"
7. "Commit changes" 클릭
```

**방법 2: Git LFS 사용**

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

#### 시나리오 4: 긴급 버그 수정

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

### 📊 업데이트 체크리스트

**배포 전 확인:**
- [ ] 로컬에서 테스트 완료 (`uv run streamlit run src/5_web_interface/app.py`)
- [ ] 변경사항을 deploy/huggingface로 동기화
- [ ] requirements.txt 최신화 (의존성 변경 시)
- [ ] 모델 파일 변경 시: Web UI로 별도 업로드 계획 확인
- [ ] 커밋 메시지 명확하게 작성

**배포 후 확인:**
- [ ] Logs에서 빌드 성공 확인
- [ ] Space URL에서 변경사항 확인 (Ctrl+Shift+R)
- [ ] 주요 기능 테스트
- [ ] 에러 로그 없는지 확인

---

## 4. 문제 해결

### 🔴 문제 1: "No module named 'ultralytics'"

**원인:** `requirements.txt`가 누락되거나 잘못됨

**해결:**
```bash
# requirements.txt 확인
cat requirements.txt
# ultralytics==8.3.229 이 있어야 함

# 재생성
uv pip compile pyproject.toml -o requirements.txt
cp requirements.txt deploy/huggingface/

# 재배포
./scripts/deploy_to_hf.sh
```

---

### 🔴 문제 2: "Model file not found"

**원인:** 모델 파일 경로 문제 또는 Git LFS 미설정

**해결:**
```bash
# 모델 파일 크기 확인 (100KB 이상이어야 함)
ls -lh models/ppe_detection/weights/best.pt

# Git LFS 설정 확인
cd safety-vision-ai
git lfs track "*.pt"
git add .gitattributes

# 모델 파일 다시 추가
git add models/ppe_detection/weights/best.pt
git commit -m "Add model file with Git LFS"
git push
```

---

### 🔴 문제 3: "Application Error" 또는 앱 시작 실패

**원인:** `app.py` 경로 또는 설정 문제

**해결:**
```bash
# 1. README.md의 app_file 확인
cat deploy/huggingface/README.md | grep app_file
# app_file: app.py 이어야 함

# 2. 파일 구조 확인
cd deploy/huggingface
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

### 🔴 문제 5: "Space is sleeping"

**원인:** 48시간 미사용 시 자동 sleep

**해결:**
- 누군가 접속하면 자동으로 깨어남 (10~30초 소요)
- 정상적인 동작이며, 무료 플랜의 한계

---

### ⚠️ 주의사항

1. **로컬 프로젝트와 deploy 동기화 필수**
   - 로컬에서 수정 → 반드시 deploy로 복사
   - deploy를 건너뛰고 직접 safety-vision-ai에서 수정하면 나중에 충돌 발생

2. **모델 파일은 별도 관리**
   - `best.pt`는 보통 150MB~600MB로 매우 큼
   - **일반 Git으로 푸시 불가** (100MB 제한)
   - **Web UI 업로드 권장**

3. **캐시 주의**
   - 브라우저 캐시 때문에 변경사항이 안 보일 수 있음
   - **Ctrl+Shift+R** (하드 리프레시) 사용

4. **빌드 실패 시**
   - Logs 탭에서 에러 확인
   - requirements.txt 의존성 문제 확인
   - 경로 문제 확인 (상대경로 사용)

5. **롤백 방법**
   ```bash
   cd safety-vision-ai
   git log  # 이전 커밋 해시 확인
   git revert <commit-hash>  # 특정 커밋 되돌리기
   git push
   ```

---

## 5. 참고 자료

### 공식 문서
- [Hugging Face Spaces 문서](https://huggingface.co/docs/hub/spaces)
- [Streamlit on Spaces 가이드](https://huggingface.co/docs/hub/spaces-sdks-streamlit)
- [Git LFS 문서](https://git-lfs.github.com/)
- [YOLOv8 문서](https://docs.ultralytics.com/)

### 커뮤니티
- [Hugging Face Discord](https://discord.com/invite/hugging-face)
- [Hugging Face Forums](https://discuss.huggingface.co/)

---

## ✅ 빠른 참조

### 주요 명령어

```bash
# 배포 (자동)
./scripts/deploy_to_hf.sh

# 로컬 테스트
uv run streamlit run src/5_web_interface/app.py

# requirements.txt 재생성
uv pip compile pyproject.toml -o requirements.txt

# 배포 상태 확인
cd safety-vision-ai
git status
git log --oneline -5
```

### 주요 경로

```
로컬 개발: src/5_web_interface/
배포 준비: deploy/huggingface/
Git 저장소: safety-vision-ai/
배포 URL: https://huggingface.co/spaces/jhboyo/safey-vision-ai
```

---

**Last Updated**: 2025-11-23
**Version**: 2.0 (간소화 버전)
**Author**: SafetyVisionAI Team
**License**: MIT
