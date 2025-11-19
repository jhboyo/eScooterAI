"""
Step 3: 데이터셋 통합

Dataset 1 (변환된 YOLO)과 Dataset 2를 하나로 통합합니다.

## 왜 통합이 필요한가?
- 두 개의 Kaggle 데이터셋을 하나의 훈련 데이터셋으로 합침
- Dataset 1: helmet만 포함 (4,581개)
- Dataset 2: helmet + vest 포함 (10,500개)
- 통합 후 총 15,081개의 이미지-라벨 쌍

## 파일명 충돌 방지
- 서로 다른 데이터셋에서 같은 파일명이 있을 수 있음
- 예: image_001.jpg가 두 데이터셋 모두에 존재
- 해결: prefix를 추가하여 구분
  - Dataset 1: ds1_image_001.jpg
  - Dataset 2: ds2_image_001.jpg

## 입력/출력
- 입력 1: dataset/raw_data/processed/dataset1/ (Step 1 결과)
- 입력 2: dataset/raw_data/raw/safety-Helmet-Reflective-Jacket/
- 출력: dataset/raw_data/processed/merged/
"""

import shutil
from pathlib import Path


def merge_datasets():
    """
    두 데이터셋을 하나로 통합

    처리 과정:
    1. 경로 설정
    2. 출력 디렉토리 생성
    3. Dataset 1 복사 (ds1_ prefix 추가)
    4. Dataset 2 복사 (ds2_ prefix 추가)
       - train, valid, test 모든 분할 포함
    5. 결과 통계 출력

    Returns:
        tuple: (총 이미지 수, 총 라벨 수)
    """

    # =========================================================================
    # 1. 경로 설정
    # =========================================================================
    # 프로젝트 루트 디렉토리
    base_dir = Path(__file__).parent.parent.parent

    # Dataset 1: Step 1에서 변환된 YOLO 데이터
    dataset1_dir = base_dir / 'dataset' / 'raw_data' / 'processed' / 'dataset1'

    # Dataset 2: 원본 YOLO 데이터 (Kaggle에서 다운로드)
    dataset2_dir = base_dir / 'dataset' / 'raw_data' / 'raw' / 'safety-Helmet-Reflective-Jacket'

    # 출력 디렉토리: 통합된 데이터 저장
    output_dir = base_dir / 'dataset' / 'raw_data' / 'processed' / 'merged'
    output_images_dir = output_dir / 'images'
    output_labels_dir = output_dir / 'labels'

    # =========================================================================
    # 2. 출력 디렉토리 생성
    # =========================================================================
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Step 3: 데이터셋 통합")
    print("=" * 50)
    print(f"Dataset 1: {dataset1_dir}")
    print(f"Dataset 2: {dataset2_dir}")
    print(f"출력 경로: {output_dir}")
    print()

    # 통계 변수
    total_images = 0
    total_labels = 0

    # =========================================================================
    # 3. Dataset 1 복사 (ds1_ prefix)
    # =========================================================================
    print("📁 Dataset 1 복사 중...")

    # 이미지와 라벨 파일 목록 가져오기
    ds1_images = list((dataset1_dir / 'images').glob('*'))
    ds1_labels = list((dataset1_dir / 'labels').glob('*.txt'))

    # 이미지 복사 (ds1_ prefix 추가)
    for img in ds1_images:
        new_name = f"ds1_{img.name}"  # 예: ds1_image_001.png
        shutil.copy(img, output_images_dir / new_name)
        total_images += 1

    # 라벨 복사 (ds1_ prefix 추가)
    for lbl in ds1_labels:
        new_name = f"ds1_{lbl.name}"  # 예: ds1_image_001.txt
        shutil.copy(lbl, output_labels_dir / new_name)
        total_labels += 1

    print(f"  - 이미지: {len(ds1_images)}개")
    print(f"  - 라벨: {len(ds1_labels)}개")

    # =========================================================================
    # 4. Dataset 2 복사 (ds2_ prefix)
    # =========================================================================
    # Dataset 2는 train/valid/test로 분할되어 있으므로 모두 합침
    print("\n📁 Dataset 2 복사 중...")
    ds2_count = {'images': 0, 'labels': 0}

    # 각 분할(train, valid, test)에 대해 처리
    for split in ['train', 'valid', 'test']:
        split_dir = dataset2_dir / split
        if not split_dir.exists():
            continue

        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'

        # -----------------------------------------------------------------
        # 이미지 복사
        # -----------------------------------------------------------------
        if images_dir.exists():
            for img in images_dir.glob('*'):
                # 이미지 파일만 처리 (jpg, jpeg, png)
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    new_name = f"ds2_{img.name}"  # 예: ds2_image_001.jpg
                    shutil.copy(img, output_images_dir / new_name)
                    ds2_count['images'] += 1
                    total_images += 1

        # -----------------------------------------------------------------
        # 라벨 복사
        # -----------------------------------------------------------------
        if labels_dir.exists():
            for lbl in labels_dir.glob('*.txt'):
                new_name = f"ds2_{lbl.name}"  # 예: ds2_image_001.txt
                shutil.copy(lbl, output_labels_dir / new_name)
                ds2_count['labels'] += 1
                total_labels += 1

    print(f"  - 이미지: {ds2_count['images']}개")
    print(f"  - 라벨: {ds2_count['labels']}개")

    # =========================================================================
    # 5. 결과 출력
    # =========================================================================
    print()
    print(f"✅ 통합 완료!")
    print(f"   - 총 이미지: {total_images}개")
    print(f"   - 총 라벨: {total_labels}개")
    print(f"   - 출력 위치: {output_dir}")
    print()

    return total_images, total_labels


if __name__ == '__main__':
    # 스크립트 직접 실행 시 통합 수행
    merge_datasets()
