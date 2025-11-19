"""
Step 2: Dataset 2 클래스 ID 확인

Dataset 2 (safety-Helmet-Reflective-Jacket)의 클래스 ID를 확인합니다.
- 이미 YOLO 형식 (.txt)
- 클래스 ID: 0 (Safety-Helmet) → helmet, 1 (Reflective-Jacket) → vest
- 변환 불필요, 확인만 수행
"""

import os
from pathlib import Path
from collections import Counter


def verify_dataset2():
    """Dataset 2 클래스 ID 확인"""

    # 경로 설정
    base_dir = Path(__file__).parent.parent.parent
    dataset2_dir = base_dir / 'images' / 'raw' / 'safety-Helmet-Reflective-Jacket'

    print("=" * 50)
    print("Step 2: Dataset 2 클래스 ID 확인")
    print("=" * 50)
    print(f"데이터셋 경로: {dataset2_dir}")
    print()

    # data.yaml 파일 확인
    yaml_path = dataset2_dir / 'data.yaml'
    if yaml_path.exists():
        print("📄 data.yaml 내용:")
        print("-" * 40)
        with open(yaml_path, 'r') as f:
            print(f.read())
        print("-" * 40)
        print()

    # 클래스 분포 확인
    total_stats = {'images': 0, 'objects': Counter()}

    print("📊 클래스 분포 분석:")
    print()

    for split in ['train', 'valid', 'test']:
        split_dir = dataset2_dir / split
        if not split_dir.exists():
            continue

        labels_dir = split_dir / 'labels'
        images_dir = split_dir / 'images'

        if not labels_dir.exists():
            continue

        # 이미지 수
        image_count = len(list(images_dir.glob('*'))) if images_dir.exists() else 0
        total_stats['images'] += image_count

        # 클래스별 객체 수
        class_count = Counter()
        for lbl_file in labels_dir.glob('*.txt'):
            with open(lbl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_count[class_id] += 1
                        total_stats['objects'][class_id] += 1

        print(f"  {split.upper()}:")
        print(f"    이미지: {image_count}개")
        print(f"    클래스 분포:")
        for class_id in sorted(class_count.keys()):
            count = class_count[class_id]
            if class_id == 0:
                class_name = "Safety-Helmet → helmet"
            elif class_id == 1:
                class_name = "Reflective-Jacket → vest"
            else:
                class_name = f"unknown_{class_id}"
            print(f"      - {class_id}: {class_name} ({count}개)")
        print()

    # 전체 통계
    print("=" * 50)
    print("📊 Dataset 2 전체 통계")
    print("=" * 50)
    print(f"총 이미지: {total_stats['images']}개")
    print(f"총 객체:")
    for class_id in sorted(total_stats['objects'].keys()):
        count = total_stats['objects'][class_id]
        if class_id == 0:
            class_name = "helmet"
        elif class_id == 1:
            class_name = "vest"
        else:
            class_name = f"unknown_{class_id}"
        print(f"   - {class_id}: {class_name} ({count}개)")
    print()

    # 클래스 매핑 확인
    print("✅ 클래스 매핑 확인:")
    print("   - 0: Safety-Helmet → 0: helmet (동일)")
    print("   - 1: Reflective-Jacket → 1: vest (동일)")
    print()
    print("📝 결론: 변환 불필요, 그대로 사용 가능")
    print()

    return total_stats


if __name__ == '__main__':
    verify_dataset2()
