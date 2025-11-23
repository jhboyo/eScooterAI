# Streamlit Cloud 배포 가이드

> Safety Vision AI - Streamlit Cloud 배포 및 관리 가이드

**배포 완료 URL**: https://safetyvisionai.streamlit.app

---

## 🎯 배포 개요

### 플랫폼 정보

| 항목 | 정보 |
|------|------|
| **플랫폼** | Streamlit Community Cloud |
| **App URL** | https://safetyvisionai.streamlit.app |
| **GitHub 리포지토리** | https://github.com/jhboyo/SafetyVisionAI |
| **브랜치** | `master` |
| **앱 파일** | `src/web_interface/app.py` |
| **Python 버전** | 3.11 |

### 주요 특징

- ✅ **자동 배포**: GitHub `master` 브랜치에 푸시하면 자동으로 배포
- ✅ **무료 호스팅**: Community Cloud는 공개 앱 무료 제공
- ✅ **자동 재시작**: 코드 변경 시 자동으로 앱 재빌드
- ✅ **로그 확인**: 실시간 로그 및 에러 확인 가능

---

## 🚀 초기 배포

### 1단계: Streamlit Cloud 연결

1. **Streamlit Cloud 접속**
   - https://share.streamlit.io/ 로그인 (GitHub 계정 사용)

2. **New App 생성**
   - "New app" 버튼 클릭
   - Repository: `jhboyo/SafetyVisionAI`
   - Branch: `master`
   - Main file path: `src/web_interface/app.py`

3. **Deploy 클릭**
   - 약 2-5분 소요
   - 자동으로 `requirements.txt` 인식 및 설치

---

## 🔄 코드 업데이트

### 기본 워크플로우

```bash
# 1. 로컬에서 코드 수정
cd /Users/joonho/workspace/sogang/tf-basic/SafetyVisionAI

# 2. 변경사항 커밋
git add .
git commit -m "Update: 변경 내용"
git push origin master

# 3. 자동 재배포 (30초~5분)
```

---

## 🐛 주요 해결 이슈

### ✅ OpenCV import 오류
- **해결**: opencv-python-headless 사용
- **파일**: requirements.txt

### ✅ 모델 경로 감지
- **해결**: Streamlit Cloud 환경 감지 로직
- **파일**: src/web_interface/utils/inference.py

### ✅ Git LFS 문제
- **해결**: 일반 Git으로 변환 (모델 파일 6MB)
- **방법**: git lfs uninstall 후 재추가

---

## 📚 참고 자료

- [Streamlit Cloud 문서](https://docs.streamlit.io/streamlit-community-cloud)
- [Deployment 가이드](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

---

**Last Updated**: 2025-11-23
