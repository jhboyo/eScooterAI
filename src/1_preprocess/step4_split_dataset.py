"""
Step 4: 데이터셋 분할

통합된 데이터셋을 Train/Val/Test로 분할합니다.

## 왜 분할이 필요한가?
머신러닝에서 데이터를 세 부분으로 나누는 것은 필수입니다:
- Train (70%): 모델 학습에 사용. 모델이 패턴을 배우는 데이터
- Val (15%): 하이퍼파라미터 튜닝, Early Stopping 판단에 사용
- Test (15%): 최종 성능 평가에 사용. 훈련 중 한 번도 보지 않은 데이터

## 분할 비율 선택 기준
- 70/15/15: 일반적인 비율 (데이터셋이 큰 경우)
- 80/10/10: 데이터셋이 작은 경우
- 60/20/20: 검증/테스트를 더 중요시하는 경우

## 랜덤 시드 (seed=42)
- 같은 시드를 사용하면 항상 같은 분할 결과가 나옴
- 재현성(reproducibility) 보장
- 42는 "은하수를 여행하는 히치하이커" 에서 유래한 관례

## 입력/출력
- 입력: dataset/raw_data/processed/merged/ (Step 3 결과)
- 출력:
  - dataset/data/train/images/, dataset/data/train/labels/
  - dataset/data/val/images/, dataset/data/val/labels/
  - dataset/data/test/images/, dataset/data/test/labels/
"""

import shutil
import random
from pathlib import Path


def split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    데이터셋을 Train/Val/Test로 분할

    처리 과정:
    1. 경로 설정 및 출력 디렉토리 생성
    2. 이미지-라벨 쌍 목록 생성
    3. 랜덤 셔플
    4. 비율에 따라 분할
    5. 각 분할로 파일 복사

    Args:
        train_ratio: 훈련 데이터 비율 (기본 0.7 = 70%)
        val_ratio: 검증 데이터 비율 (기본 0.15 = 15%)
        test_ratio: 테스트 데이터 비율 (기본 0.15 = 15%)
        seed: 랜덤 시드 (재현성 보장, 기본 42)

    Returns:
        dict: 각 분할의 파일 stem 목록
              {'train': [...], 'val': [...], 'test': [...]}
    """
    # 랜덤 시드 설정 (재현성 보장)
    random.seed(seed)

    # =========================================================================
    # 1. 경로 설정
    # =========================================================================
    # 프로젝트 루트 디렉토리
    base_dir = Path(__file__).parent.parent.parent

    # 입력 디렉토리: Step 3에서 통합된 데이터
    merged_dir = base_dir / 'dataset' / 'raw_data' / 'processed' / 'merged'
    merged_images = merged_dir / 'images'
    merged_labels = merged_dir / 'labels'

    # 출력 디렉토리: 최종 분할된 데이터
    output_base = base_dir / 'dataset' / 'data'

    # 각 분할의 출력 경로 정의
    splits = {
        'train': output_base / 'train',  # 훈련 데이터
        'val': output_base / 'val',      # 검증 데이터
        'test': output_base / 'test'     # 테스트 데이터
    }

    print("=" * 50)
    print("Step 4: 데이터셋 분할")
    print("=" * 50)
    print(f"입력 경로: {merged_dir}")
    print(f"분할 비율: Train {train_ratio*100:.0f}% / Val {val_ratio*100:.0f}% / Test {test_ratio*100:.0f}%")
    print()

    # =========================================================================
    # 2. 출력 디렉토리 생성 (기존 파일 삭제)
    # =========================================================================
    for split_name, split_dir in splits.items():
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'

        # 기존 파일 삭제 (이전 실행 결과 제거)
        if images_dir.exists():
            shutil.rmtree(images_dir)
        if labels_dir.exists():
            shutil.rmtree(labels_dir)

        # 새로 생성
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 3. 이미지-라벨 쌍 목록 생성
    # =========================================================================
    # 라벨이 있는 이미지만 포함 (orphan 이미지 제외)
    all_images = []

    for img_path in merged_images.glob('*'):
        # 이미지 파일만 처리
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # 대응하는 라벨 파일 확인
            label_path = merged_labels / (img_path.stem + '.txt')
            if label_path.exists():
                # stem: 파일명에서 확장자를 뺀 부분
                # 예: ds1_image_001.png → ds1_image_001
                all_images.append(img_path.stem)

    # =========================================================================
    # 4. 랜덤 셔플 및 분할
    # =========================================================================
    # 데이터 순서를 랜덤하게 섞음
    random.shuffle(all_images)

    # 분할 인덱스 계산
    total = len(all_images)
    train_end = int(total * train_ratio)           # Train 끝 인덱스
    val_end = train_end + int(total * val_ratio)   # Val 끝 인덱스

    # 분할 수행
    split_data = {
        'train': all_images[:train_end],           # 0 ~ train_end
        'val': all_images[train_end:val_end],      # train_end ~ val_end
        'test': all_images[val_end:]               # val_end ~ 끝
    }

    print(f"총 데이터: {total}개")
    print()

    # =========================================================================
    # 5. 파일 복사
    # =========================================================================
    for split_name, file_stems in split_data.items():
        split_dir = splits[split_name]
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'

        # 각 파일에 대해 이미지와 라벨 복사
        for stem in file_stems:
            # ---------------------------------------------------------
            # 이미지 복사 (확장자가 다를 수 있으므로 확인)
            # ---------------------------------------------------------
            for ext in ['.jpg', '.jpeg', '.png']:
                src_img = merged_images / (stem + ext)
                if src_img.exists():
                    shutil.copy(src_img, images_dir / src_img.name)
                    break

            # ---------------------------------------------------------
            # 라벨 복사
            # ---------------------------------------------------------
            src_label = merged_labels / (stem + '.txt')
            if src_label.exists():
                shutil.copy(src_label, labels_dir / src_label.name)

        print(f"  {split_name}: {len(file_stems)}개")

    # =========================================================================
    # 6. 결과 출력
    # =========================================================================
    print()
    print(f"✅ 분할 완료!")
    print(f"   - Train: {len(split_data['train'])}개 ({len(split_data['train'])*100//total}%)")
    print(f"   - Val: {len(split_data['val'])}개 ({len(split_data['val'])*100//total}%)")
    print(f"   - Test: {len(split_data['test'])}개 ({len(split_data['test'])*100//total}%)")
    print()

    return split_data


if __name__ == '__main__':
    # 스크립트 직접 실행 시 분할 수행
    split_dataset()
