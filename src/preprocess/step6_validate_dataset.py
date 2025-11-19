"""
Step 5: 데이터셋 검증

전처리된 데이터셋의 품질을 검증합니다.
- 이미지-라벨 매칭 확인
- 클래스 분포 분석
- 샘플 이미지 시각화
"""

import os
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

# 클래스 정의
CLASS_NAMES = {0: 'helmet', 1: 'vest'}
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 165, 0)}  # BGR: 초록, 주황


def validate_dataset():
    """데이터셋 검증"""

    # 경로 설정
    base_dir = Path(__file__).parent.parent.parent
    images_dir = base_dir / 'images'

    print("=" * 50)
    print("Step 5: 데이터셋 검증")
    print("=" * 50)
    print()

    splits = ['train', 'val', 'test']
    total_stats = {'images': 0, 'labels': 0, 'objects': Counter()}

    for split in splits:
        split_dir = images_dir / split
        img_dir = split_dir / 'images'
        lbl_dir = split_dir / 'labels'

        if not split_dir.exists():
            print(f"⚠️  {split} 폴더가 없습니다.")
            continue

        # 파일 목록
        images = set(p.stem for p in img_dir.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png'])
        labels = set(p.stem for p in lbl_dir.glob('*.txt'))

        # 매칭 확인
        matched = images & labels
        img_only = images - labels
        lbl_only = labels - images

        print(f"📁 {split.upper()}")
        print(f"   이미지: {len(images)}개")
        print(f"   라벨: {len(labels)}개")
        print(f"   매칭됨: {len(matched)}개")

        if img_only:
            print(f"   ⚠️  라벨 없는 이미지: {len(img_only)}개")
        if lbl_only:
            print(f"   ⚠️  이미지 없는 라벨: {len(lbl_only)}개")

        # 클래스 분포 분석
        class_count = Counter()
        for lbl_file in lbl_dir.glob('*.txt'):
            with open(lbl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_count[class_id] += 1
                        total_stats['objects'][class_id] += 1

        print(f"   클래스 분포:")
        for class_id, count in sorted(class_count.items()):
            class_name = CLASS_NAMES.get(class_id, f'unknown_{class_id}')
            print(f"      - {class_name}: {count}개")

        total_stats['images'] += len(matched)
        total_stats['labels'] += len(labels)
        print()

    # 전체 통계
    print("=" * 50)
    print("📊 전체 통계")
    print("=" * 50)
    print(f"총 이미지: {total_stats['images']}개")
    print(f"총 객체:")
    for class_id, count in sorted(total_stats['objects'].items()):
        class_name = CLASS_NAMES.get(class_id, f'unknown_{class_id}')
        print(f"   - {class_name}: {count}개")
    print()

    return total_stats


def visualize_samples(num_samples=5):
    """샘플 이미지 시각화"""

    # 경로 설정
    base_dir = Path(__file__).parent.parent.parent
    images_dir = base_dir / 'images'
    output_dir = base_dir / 'images' / 'processed' / 'samples'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("🖼️  샘플 이미지 시각화")
    print("=" * 50)

    train_images = list((images_dir / 'train' / 'images').glob('*'))

    if not train_images:
        print("⚠️  훈련 이미지가 없습니다.")
        return

    # 랜덤 샘플 선택
    import random
    samples = random.sample(train_images, min(num_samples, len(train_images)))

    for i, img_path in enumerate(samples):
        # 이미지 읽기
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # 라벨 읽기
        lbl_path = images_dir / 'train' / 'labels' / (img_path.stem + '.txt')
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * w
                        y_center = float(parts[2]) * h
                        box_w = float(parts[3]) * w
                        box_h = float(parts[4]) * h

                        # 바운딩 박스 좌표
                        x1 = int(x_center - box_w / 2)
                        y1 = int(y_center - box_h / 2)
                        x2 = int(x_center + box_w / 2)
                        y2 = int(y_center + box_h / 2)

                        # 색상
                        color = CLASS_COLORS.get(class_id, (255, 255, 255))
                        class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')

                        # 바운딩 박스 그리기
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, class_name, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 저장
        output_path = output_dir / f'sample_{i+1}.jpg'
        cv2.imwrite(str(output_path), img)
        print(f"  저장: {output_path}")

    print()
    print(f"✅ {len(samples)}개 샘플 이미지 저장 완료!")
    print(f"   - 출력 위치: {output_dir}")
    print()


if __name__ == '__main__':
    validate_dataset()
    visualize_samples()
