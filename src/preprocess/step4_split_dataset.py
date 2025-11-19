"""
Step 3: 데이터셋 분할

통합된 데이터셋을 Train/Val/Test로 분할합니다.
- Train: 70%
- Val: 15%
- Test: 15%
"""

import os
import shutil
import random
from pathlib import Path


def split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    데이터셋을 Train/Val/Test로 분할

    Args:
        train_ratio: 훈련 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        seed: 랜덤 시드
    """
    random.seed(seed)

    # 경로 설정
    base_dir = Path(__file__).parent.parent.parent

    # 입력 디렉토리 (통합된 데이터)
    merged_dir = base_dir / 'images' / 'processed' / 'merged'
    merged_images = merged_dir / 'images'
    merged_labels = merged_dir / 'labels'

    # 출력 디렉토리
    output_base = base_dir / 'images'

    splits = {
        'train': output_base / 'train',
        'val': output_base / 'val',
        'test': output_base / 'test'
    }

    print("=" * 50)
    print("Step 3: 데이터셋 분할")
    print("=" * 50)
    print(f"입력 경로: {merged_dir}")
    print(f"분할 비율: Train {train_ratio*100:.0f}% / Val {val_ratio*100:.0f}% / Test {test_ratio*100:.0f}%")
    print()

    # 출력 디렉토리 생성 및 기존 파일 삭제
    for split_name, split_dir in splits.items():
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'

        # 기존 파일 삭제
        if images_dir.exists():
            shutil.rmtree(images_dir)
        if labels_dir.exists():
            shutil.rmtree(labels_dir)

        # 새로 생성
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 파일 목록 (라벨이 있는 것만)
    all_images = []
    for img_path in merged_images.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # 대응하는 라벨 파일 확인
            label_path = merged_labels / (img_path.stem + '.txt')
            if label_path.exists():
                all_images.append(img_path.stem)

    # 셔플
    random.shuffle(all_images)

    # 분할
    total = len(all_images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    split_data = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }

    print(f"총 데이터: {total}개")
    print()

    # 파일 복사
    for split_name, file_stems in split_data.items():
        split_dir = splits[split_name]
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'

        for stem in file_stems:
            # 이미지 복사
            for ext in ['.jpg', '.jpeg', '.png']:
                src_img = merged_images / (stem + ext)
                if src_img.exists():
                    shutil.copy(src_img, images_dir / src_img.name)
                    break

            # 라벨 복사
            src_label = merged_labels / (stem + '.txt')
            if src_label.exists():
                shutil.copy(src_label, labels_dir / src_label.name)

        print(f"  {split_name}: {len(file_stems)}개")

    print()
    print(f"✅ 분할 완료!")
    print(f"   - Train: {len(split_data['train'])}개 ({len(split_data['train'])*100//total}%)")
    print(f"   - Val: {len(split_data['val'])}개 ({len(split_data['val'])*100//total}%)")
    print(f"   - Test: {len(split_data['test'])}개 ({len(split_data['test'])*100//total}%)")
    print()

    return split_data


if __name__ == '__main__':
    split_dataset()
