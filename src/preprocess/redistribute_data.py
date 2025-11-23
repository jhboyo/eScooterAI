"""
데이터 재분배 스크립트

Train 데이터를 특정 크기로 조정하고 나머지를 Val, Test로 균등 분배합니다.

## 사용 목적
Step 4에서 초기 분할(70/15/15) 후, Train 데이터를 제한하여 검증 데이터를 확보

## 실행 결과 (3 Class)
- 초기: Train 10,850 / Val 2,325 / Test 2,325
- 재분배 후: Train 9,999 / Val 2,750 / Test 2,751
- 비율: 64.5% / 17.7% / 17.7%

## 사용 방법
```bash
uv run python src/preprocess/redistribute_data.py
```

## 파라미터 커스터마이징
스크립트 내 target_train_size를 원하는 값으로 변경 (기본값: 9999)

## 주의사항
- 이미지와 라벨이 함께 이동됩니다
- 랜덤하게 선택되므로 seed=42로 재현성 보장
- 실행 전 백업 권장
"""

import shutil
import random
from pathlib import Path


def redistribute_data(target_train_size=9999, seed=42):
    """
    Train 데이터를 줄이고 Val, Test로 재분배

    Args:
        target_train_size: 목표 train 이미지 수
        seed: 랜덤 시드
    """
    # 경로 설정
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / 'dataset' / 'data'

    train_images_dir = data_dir / 'train' / 'images'
    train_labels_dir = data_dir / 'train' / 'labels'
    val_images_dir = data_dir / 'val' / 'images'
    val_labels_dir = data_dir / 'val' / 'labels'
    test_images_dir = data_dir / 'test' / 'images'
    test_labels_dir = data_dir / 'test' / 'labels'

    print("=" * 50)
    print("데이터 재분배")
    print("=" * 50)

    # 현재 상태 확인
    train_images = list(train_images_dir.glob('*'))
    val_images = list(val_images_dir.glob('*'))
    test_images = list(test_images_dir.glob('*'))

    current_train = len(train_images)
    current_val = len(val_images)
    current_test = len(test_images)

    print(f"\n현재 분포:")
    print(f"  Train: {current_train}개")
    print(f"  Val: {current_val}개")
    print(f"  Test: {current_test}개")
    print(f"  총: {current_train + current_val + current_test}개")

    # 이동할 이미지 수 계산
    to_move = current_train - target_train_size

    if to_move <= 0:
        print(f"\n이미 Train이 {current_train}개입니다. 재분배가 필요 없습니다.")
        return

    # Val과 Test에 균등 분배
    to_val = to_move // 2
    to_test = to_move - to_val

    print(f"\n이동할 이미지:")
    print(f"  Train → Val: {to_val}개")
    print(f"  Train → Test: {to_test}개")
    print(f"  총 이동: {to_move}개")

    # 랜덤하게 이미지 선택
    random.seed(seed)
    random.shuffle(train_images)

    # 이동할 이미지 선택
    to_val_images = train_images[:to_val]
    to_test_images = train_images[to_val:to_val + to_test]

    # Val로 이동
    print(f"\nVal로 이동 중... ({to_val}개)")
    for img in to_val_images:
        # 이미지 파일명 (확장자 포함)
        img_name = img.name
        # 라벨 파일명 (확장자를 .txt로 변경)
        label_name = img.stem + '.txt'

        # 이미지 이동
        src_img = train_images_dir / img_name
        dst_img = val_images_dir / img_name
        shutil.move(src_img, dst_img)

        # 라벨 이동
        src_label = train_labels_dir / label_name
        dst_label = val_labels_dir / label_name
        if src_label.exists():
            shutil.move(src_label, dst_label)

    # Test로 이동
    print(f"Test로 이동 중... ({to_test}개)")
    for img in to_test_images:
        img_name = img.name
        label_name = img.stem + '.txt'

        # 이미지 이동
        src_img = train_images_dir / img_name
        dst_img = test_images_dir / img_name
        shutil.move(src_img, dst_img)

        # 라벨 이동
        src_label = train_labels_dir / label_name
        dst_label = test_labels_dir / label_name
        if src_label.exists():
            shutil.move(src_label, dst_label)

    # 최종 결과 확인
    final_train = len(list(train_images_dir.glob('*')))
    final_val = len(list(val_images_dir.glob('*')))
    final_test = len(list(test_images_dir.glob('*')))

    print("\n" + "=" * 50)
    print("✅ 재분배 완료!")
    print("=" * 50)
    print(f"\n최종 분포:")
    print(f"  Train: {final_train}개 ({final_train / (final_train + final_val + final_test) * 100:.1f}%)")
    print(f"  Val: {final_val}개 ({final_val / (final_train + final_val + final_test) * 100:.1f}%)")
    print(f"  Test: {final_test}개 ({final_test / (final_train + final_val + final_test) * 100:.1f}%)")
    print(f"  총: {final_train + final_val + final_test}개")
    print()


if __name__ == '__main__':
    redistribute_data(target_train_size=9999)
