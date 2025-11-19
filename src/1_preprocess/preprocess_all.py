"""
전체 전처리 파이프라인

모든 전처리 단계를 순차적으로 실행합니다.

사용법:
    uv run python src/1_preprocess/preprocess_all.py
"""

import time

from step1_convert_voc_to_yolo import convert_dataset1
from step2_verify_dataset2 import verify_dataset2
from step3_merge_datasets import merge_datasets
from step4_split_dataset import split_dataset
from step5_generate_yaml import generate_yaml
from step6_validate_dataset import validate_dataset, visualize_samples


def run_all():
    """전체 전처리 파이프라인 실행"""

    print()
    print("=" * 60)
    print("🚀 PPE Detection 데이터셋 전처리 시작")
    print("=" * 60)
    print()

    start_time = time.time()

    try:
        # Step 1: VOC → YOLO 변환
        print("\n" + "=" * 60)
        convert_dataset1()

        # Step 2: Dataset 2 클래스 ID 확인
        print("\n" + "=" * 60)
        verify_dataset2()

        # Step 3: 데이터셋 통합
        print("\n" + "=" * 60)
        merge_datasets()

        # Step 4: Train/Val/Test 분할
        print("\n" + "=" * 60)
        split_dataset()

        # Step 5: YAML 파일 생성
        print("\n" + "=" * 60)
        generate_yaml()

        # Step 6: 데이터셋 검증
        print("\n" + "=" * 60)
        validate_dataset()
        visualize_samples()

        elapsed_time = time.time() - start_time

        print()
        print("=" * 60)
        print("✅ 전처리 완료!")
        print("=" * 60)
        print(f"소요 시간: {elapsed_time:.1f}초")
        print()
        print("📁 생성된 파일:")
        print("   - images/train/       (훈련 데이터)")
        print("   - images/val/         (검증 데이터)")
        print("   - images/test/        (테스트 데이터)")
        print("   - configs/ppe_dataset.yaml (데이터셋 설정)")
        print()
        print("다음 단계:")
        print("   uv run python src/2_training/train.py --data configs/ppe_dataset.yaml")
        print()

    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ 전처리 실패: {str(e)}")
        print("=" * 60)
        raise


if __name__ == '__main__':
    run_all()
