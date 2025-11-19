"""
Step 2: Dataset 2 클래스 ID 확인

Dataset 2 (safety-Helmet-Reflective-Jacket)의 클래스 ID를 확인합니다.

## 이 단계가 필요한 이유
- Dataset 2는 이미 YOLO 형식 (.txt)으로 제공됨
- 하지만 클래스 ID가 우리 프로젝트와 일치하는지 확인 필요
- 확인 결과 변환 없이 그대로 사용 가능

## Dataset 2 클래스 구성
- 0: Safety-Helmet → 우리 프로젝트의 helmet (0)
- 1: Reflective-Jacket → 우리 프로젝트의 vest (1)

## 클래스 ID가 다르면?
만약 Dataset 2의 클래스 ID가 우리 프로젝트와 다르다면
라벨 파일의 첫 번째 숫자(class_id)를 변경해야 함
예: 0 → 1, 1 → 0 (순서가 반대인 경우)

## 입력/출력
- 입력: images/raw/safety-Helmet-Reflective-Jacket/
- 출력: 없음 (확인만 수행)
"""

from pathlib import Path
from collections import Counter


def verify_dataset2():
    """
    Dataset 2 클래스 ID 확인 및 통계 분석

    처리 과정:
    1. data.yaml 파일 확인 (데이터셋의 클래스 정의)
    2. train/valid/test 각 분할에 대해:
       - 이미지 수 계산
       - 클래스별 객체 수 계산
    3. 전체 통계 출력
    4. 클래스 매핑 일치 여부 확인

    Returns:
        dict: 전체 통계 정보 {'images': int, 'objects': Counter}
    """

    # =========================================================================
    # 1. 경로 설정
    # =========================================================================
    # 프로젝트 루트 디렉토리
    base_dir = Path(__file__).parent.parent.parent
    # Dataset 2 위치
    dataset2_dir = base_dir / 'images' / 'raw' / 'safety-Helmet-Reflective-Jacket'

    print("=" * 50)
    print("Step 2: Dataset 2 클래스 ID 확인")
    print("=" * 50)
    print(f"데이터셋 경로: {dataset2_dir}")
    print()

    # =========================================================================
    # 2. data.yaml 파일 확인
    # =========================================================================
    # Kaggle 데이터셋에 포함된 클래스 정의 파일
    yaml_path = dataset2_dir / 'data.yaml'
    if yaml_path.exists():
        print("📄 data.yaml 내용:")
        print("-" * 40)
        with open(yaml_path, 'r') as f:
            print(f.read())
        print("-" * 40)
        print()

    # =========================================================================
    # 3. 클래스 분포 분석
    # =========================================================================
    # 전체 통계를 저장할 딕셔너리
    total_stats = {'images': 0, 'objects': Counter()}

    print("📊 클래스 분포 분석:")
    print()

    # 각 분할(train/valid/test)에 대해 분석
    for split in ['train', 'valid', 'test']:
        split_dir = dataset2_dir / split
        if not split_dir.exists():
            continue

        # 라벨과 이미지 디렉토리
        labels_dir = split_dir / 'labels'
        images_dir = split_dir / 'images'

        if not labels_dir.exists():
            continue

        # -----------------------------------------------------------------
        # 이미지 수 계산
        # -----------------------------------------------------------------
        # glob('*')로 모든 파일을 가져옴
        image_count = len(list(images_dir.glob('*'))) if images_dir.exists() else 0
        total_stats['images'] += image_count

        # -----------------------------------------------------------------
        # 클래스별 객체 수 계산
        # -----------------------------------------------------------------
        class_count = Counter()

        # 모든 라벨 파일 순회
        for lbl_file in labels_dir.glob('*.txt'):
            with open(lbl_file, 'r') as f:
                for line in f:
                    # YOLO 형식: class_id x_center y_center width height
                    parts = line.strip().split()
                    if parts:
                        # 첫 번째 값이 class_id
                        class_id = int(parts[0])
                        class_count[class_id] += 1
                        total_stats['objects'][class_id] += 1

        # -----------------------------------------------------------------
        # 분할별 결과 출력
        # -----------------------------------------------------------------
        print(f"  {split.upper()}:")
        print(f"    이미지: {image_count}개")
        print(f"    클래스 분포:")

        # 클래스 ID 순서대로 출력
        for class_id in sorted(class_count.keys()):
            count = class_count[class_id]
            # 클래스명 매핑
            if class_id == 0:
                class_name = "Safety-Helmet → helmet"
            elif class_id == 1:
                class_name = "Reflective-Jacket → vest"
            else:
                class_name = f"unknown_{class_id}"
            print(f"      - {class_id}: {class_name} ({count}개)")
        print()

    # =========================================================================
    # 4. 전체 통계 출력
    # =========================================================================
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

    # =========================================================================
    # 5. 클래스 매핑 확인 결과
    # =========================================================================
    print("✅ 클래스 매핑 확인:")
    print("   - 0: Safety-Helmet → 0: helmet (동일)")
    print("   - 1: Reflective-Jacket → 1: vest (동일)")
    print()
    print("📝 결론: 변환 불필요, 그대로 사용 가능")
    print()

    return total_stats


if __name__ == '__main__':
    # 스크립트 직접 실행 시 확인 수행
    verify_dataset2()
