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

# 모델 파일 업데이트 확인
echo -e "${YELLOW}모델 파일(best.pt)을 업데이트하시겠습니까? (y/N):${NC}"
read -p "Update model? " UPDATE_MODEL

if [[ "$UPDATE_MODEL" =~ ^[Yy]$ ]]; then
    echo -e "${RED}⚠️  모델 파일은 크기가 크므로 Git LFS 또는 Web UI 사용을 권장합니다.${NC}"
    echo -e "${YELLOW}선택하세요:${NC}"
    echo "  1) Web UI로 수동 업로드 (권장)"
    echo "  2) Git LFS로 푸시 (고급)"
    echo "  3) 건너뛰기"
    read -p "선택 (1-3): " MODEL_CHOICE

    case $MODEL_CHOICE in
        1)
            echo -e "${YELLOW}모델 파일을 Web UI로 업로드하세요:${NC}"
            echo "  1. https://huggingface.co/spaces/jhboyo/safety-vision-ai/tree/main"
            echo "  2. models/ppe_detection/weights/ 로 이동"
            echo "  3. 'Upload files' 클릭하여 best.pt 업로드"
            echo ""
            read -p "업로드 완료 후 Enter를 눌러 계속..."
            ;;
        2)
            echo -e "${YELLOW}모델 파일을 복사합니다...${NC}"
            mkdir -p "$DEPLOY_DIR/models/ppe_detection/weights"
            cp "$PROJECT_ROOT/models/ppe_detection/weights/best.pt" "$DEPLOY_DIR/models/ppe_detection/weights/"
            echo -e "${GREEN}✓ 모델 파일 복사 완료 (Git LFS 필요)${NC}"
            ;;
        *)
            echo -e "${YELLOW}모델 파일 업데이트 건너뛰기${NC}"
            ;;
    esac
fi

# Step 1: 로컬 → deploy 동기화
echo -e "${YELLOW}Step 1: 로컬 변경사항을 deploy 디렉토리로 동기화...${NC}"
rsync -av --delete \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='models/ppe_detection/weights/best.pt' \
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
