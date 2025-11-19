"""
Step 2: 데이터셋 통합

Dataset 1 (변환된 YOLO)과 Dataset 2를 하나로 통합합니다.
- 파일명에 prefix 추가하여 충돌 방지
- Dataset 1: ds1_ prefix
- Dataset 2: ds2_ prefix
"""

import os
import shutil
from pathlib import Path


def merge_datasets():
    """두 데이터셋을 하나로 통합"""

    # 경로 설정
    base_dir = Path(__file__).parent.parent.parent

    # Dataset 1 (변환된 YOLO)
    dataset1_dir = base_dir / 'images' / 'processed' / 'dataset1'

    # Dataset 2 (원본 YOLO)
    dataset2_dir = base_dir / 'images' / 'raw' / 'safety-Helmet-Reflective-Jacket'

    # 출력 디렉토리
    output_dir = base_dir / 'images' / 'processed' / 'merged'
    output_images_dir = output_dir / 'images'
    output_labels_dir = output_dir / 'labels'

    # 출력 디렉토리 생성
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Step 2: 데이터셋 통합")
    print("=" * 50)
    print(f"Dataset 1: {dataset1_dir}")
    print(f"Dataset 2: {dataset2_dir}")
    print(f"출력 경로: {output_dir}")
    print()

    total_images = 0
    total_labels = 0

    # Dataset 1 복사 (ds1_ prefix)
    print("📁 Dataset 1 복사 중...")
    ds1_images = list((dataset1_dir / 'images').glob('*'))
    ds1_labels = list((dataset1_dir / 'labels').glob('*.txt'))

    for img in ds1_images:
        new_name = f"ds1_{img.name}"
        shutil.copy(img, output_images_dir / new_name)
        total_images += 1

    for lbl in ds1_labels:
        new_name = f"ds1_{lbl.name}"
        shutil.copy(lbl, output_labels_dir / new_name)
        total_labels += 1

    print(f"  - 이미지: {len(ds1_images)}개")
    print(f"  - 라벨: {len(ds1_labels)}개")

    # Dataset 2 복사 (ds2_ prefix) - train, valid, test 모두 포함
    print("\n📁 Dataset 2 복사 중...")
    ds2_count = {'images': 0, 'labels': 0}

    for split in ['train', 'valid', 'test']:
        split_dir = dataset2_dir / split
        if not split_dir.exists():
            continue

        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'

        if images_dir.exists():
            for img in images_dir.glob('*'):
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    new_name = f"ds2_{img.name}"
                    shutil.copy(img, output_images_dir / new_name)
                    ds2_count['images'] += 1
                    total_images += 1

        if labels_dir.exists():
            for lbl in labels_dir.glob('*.txt'):
                new_name = f"ds2_{lbl.name}"
                shutil.copy(lbl, output_labels_dir / new_name)
                ds2_count['labels'] += 1
                total_labels += 1

    print(f"  - 이미지: {ds2_count['images']}개")
    print(f"  - 라벨: {ds2_count['labels']}개")

    print()
    print(f"✅ 통합 완료!")
    print(f"   - 총 이미지: {total_images}개")
    print(f"   - 총 라벨: {total_labels}개")
    print(f"   - 출력 위치: {output_dir}")
    print()

    return total_images, total_labels


if __name__ == '__main__':
    merge_datasets()
