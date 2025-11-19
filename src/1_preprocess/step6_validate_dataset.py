"""
Step 6: 데이터셋 검증

전처리된 데이터셋의 품질을 검증합니다.

## 왜 검증이 필요한가?
전처리 후 데이터셋에 문제가 없는지 확인해야 합니다:
- 이미지와 라벨이 1:1로 매칭되는가?
- 라벨 파일이 누락된 이미지는 없는가?
- 클래스 분포가 적절한가? (불균형이 심하면 학습에 영향)
- 바운딩 박스가 올바르게 그려지는가?

## 검증 항목
1. **이미지-라벨 매칭 확인**
   - 이미지는 있는데 라벨이 없는 경우 (orphan image)
   - 라벨은 있는데 이미지가 없는 경우 (orphan label)

2. **클래스 분포 분석**
   - 각 클래스별 객체 수 계산
   - Train/Val/Test 분포 확인

3. **시각화**
   - 랜덤 샘플에 바운딩 박스를 그려서 확인
   - 라벨이 올바르게 적용되었는지 육안 검증

## 클래스 불균형
- helmet: 39,157개, vest: 16,049개 (약 2.4:1)
- 불균형이 있지만 심각하지 않음
- 필요시 데이터 증강이나 클래스 가중치로 해결 가능

## 입력/출력
- 입력: dataset/data/train/, dataset/data/val/, dataset/data/test/
- 출력: dataset/raw_data/processed/samples/ (시각화 이미지)
"""

from pathlib import Path
from collections import Counter
import cv2

# =============================================================================
# 클래스 정의
# =============================================================================
# 클래스 ID와 이름 매핑
CLASS_NAMES = {0: 'helmet', 1: 'vest'}

# 시각화 시 사용할 색상 (BGR 형식)
# 초록색: helmet, 주황색: vest
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 165, 0)}


def validate_dataset():
    """
    데이터셋 검증

    처리 과정:
    1. 각 분할(train/val/test)에 대해:
       - 이미지와 라벨 파일 목록 추출
       - 매칭 여부 확인
       - 클래스별 객체 수 계산
    2. 전체 통계 출력

    Returns:
        dict: 전체 통계 정보
              {'images': int, 'labels': int, 'objects': Counter}
    """

    # =========================================================================
    # 1. 경로 설정
    # =========================================================================
    base_dir = Path(__file__).parent.parent.parent
    images_dir = base_dir / 'dataset' / 'data'

    print("=" * 50)
    print("Step 6: 데이터셋 검증")
    print("=" * 50)
    print()

    # 검증할 분할 목록
    splits = ['train', 'val', 'test']
    # 전체 통계를 저장할 딕셔너리
    total_stats = {'images': 0, 'labels': 0, 'objects': Counter()}

    # =========================================================================
    # 2. 각 분할에 대해 검증
    # =========================================================================
    for split in splits:
        split_dir = images_dir / split
        img_dir = split_dir / 'images'
        lbl_dir = split_dir / 'labels'

        # 폴더가 없으면 스킵
        if not split_dir.exists():
            print(f"⚠️  {split} 폴더가 없습니다.")
            continue

        # -----------------------------------------------------------------
        # 파일 목록 추출
        # -----------------------------------------------------------------
        # stem: 파일명에서 확장자를 뺀 부분
        # 예: image_001.jpg → image_001
        images = set(
            p.stem for p in img_dir.glob('*')
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
        )
        labels = set(p.stem for p in lbl_dir.glob('*.txt'))

        # -----------------------------------------------------------------
        # 매칭 확인
        # -----------------------------------------------------------------
        # 교집합: 이미지와 라벨이 모두 있는 파일
        matched = images & labels
        # 차집합: 라벨이 없는 이미지 (orphan image)
        img_only = images - labels
        # 차집합: 이미지가 없는 라벨 (orphan label)
        lbl_only = labels - images

        print(f"📁 {split.upper()}")
        print(f"   이미지: {len(images)}개")
        print(f"   라벨: {len(labels)}개")
        print(f"   매칭됨: {len(matched)}개")

        # 경고 출력
        if img_only:
            print(f"   ⚠️  라벨 없는 이미지: {len(img_only)}개")
        if lbl_only:
            print(f"   ⚠️  이미지 없는 라벨: {len(lbl_only)}개")

        # -----------------------------------------------------------------
        # 클래스 분포 분석
        # -----------------------------------------------------------------
        class_count = Counter()

        # 모든 라벨 파일 순회
        for lbl_file in lbl_dir.glob('*.txt'):
            with open(lbl_file, 'r') as f:
                for line in f:
                    # YOLO 형식: class_id x_center y_center width height
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_count[class_id] += 1
                        total_stats['objects'][class_id] += 1

        # 클래스별 결과 출력
        print(f"   클래스 분포:")
        for class_id, count in sorted(class_count.items()):
            class_name = CLASS_NAMES.get(class_id, f'unknown_{class_id}')
            print(f"      - {class_name}: {count}개")

        # 통계 누적
        total_stats['images'] += len(matched)
        total_stats['labels'] += len(labels)
        print()

    # =========================================================================
    # 3. 전체 통계 출력
    # =========================================================================
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
    """
    샘플 이미지에 바운딩 박스를 그려서 시각화

    처리 과정:
    1. 훈련 이미지에서 랜덤 샘플 선택
    2. 각 샘플에 대해:
       - 이미지 로드
       - 대응하는 라벨 파일 읽기
       - YOLO 좌표를 픽셀 좌표로 변환
       - 바운딩 박스와 클래스명 그리기
    3. 결과 이미지 저장

    Args:
        num_samples: 시각화할 샘플 수 (기본 5개)
    """

    # =========================================================================
    # 1. 경로 설정
    # =========================================================================
    base_dir = Path(__file__).parent.parent.parent
    images_dir = base_dir / 'dataset' / 'data'
    # 시각화 결과를 저장할 디렉토리
    output_dir = base_dir / 'dataset' / 'raw_data' / 'processed' / 'samples'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("🖼️  샘플 이미지 시각화")
    print("=" * 50)

    # 훈련 이미지 목록
    train_images = list((images_dir / 'train' / 'images').glob('*'))

    if not train_images:
        print("⚠️  훈련 이미지가 없습니다.")
        return

    # =========================================================================
    # 2. 랜덤 샘플 선택
    # =========================================================================
    import random
    samples = random.sample(train_images, min(num_samples, len(train_images)))

    # =========================================================================
    # 3. 각 샘플에 대해 시각화
    # =========================================================================
    for i, img_path in enumerate(samples):
        # -----------------------------------------------------------------
        # 이미지 로드
        # -----------------------------------------------------------------
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 이미지 크기 (YOLO 좌표 변환에 필요)
        h, w = img.shape[:2]

        # -----------------------------------------------------------------
        # 라벨 파일 읽기 및 바운딩 박스 그리기
        # -----------------------------------------------------------------
        lbl_path = images_dir / 'train' / 'labels' / (img_path.stem + '.txt')
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO 형식 파싱
                        class_id = int(parts[0])
                        x_center = float(parts[1])  # 정규화된 값 (0~1)
                        y_center = float(parts[2])
                        box_w = float(parts[3])
                        box_h = float(parts[4])

                        # ---------------------------------------------
                        # YOLO 좌표 → 픽셀 좌표 변환
                        # ---------------------------------------------
                        # 정규화된 좌표를 픽셀 좌표로 변환
                        x_center_px = x_center * w
                        y_center_px = y_center * h
                        box_w_px = box_w * w
                        box_h_px = box_h * h

                        # 바운딩 박스의 좌상단/우하단 좌표 계산
                        x1 = int(x_center_px - box_w_px / 2)
                        y1 = int(y_center_px - box_h_px / 2)
                        x2 = int(x_center_px + box_w_px / 2)
                        y2 = int(y_center_px + box_h_px / 2)

                        # ---------------------------------------------
                        # 바운딩 박스 그리기
                        # ---------------------------------------------
                        color = CLASS_COLORS.get(class_id, (255, 255, 255))
                        class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')

                        # 박스 그리기 (두께 2)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        # 클래스명 텍스트
                        cv2.putText(img, class_name, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # -----------------------------------------------------------------
        # 결과 저장
        # -----------------------------------------------------------------
        output_path = output_dir / f'sample_{i+1}.jpg'
        cv2.imwrite(str(output_path), img)
        print(f"  저장: {output_path}")

    # =========================================================================
    # 4. 결과 출력
    # =========================================================================
    print()
    print(f"✅ {len(samples)}개 샘플 이미지 저장 완료!")
    print(f"   - 출력 위치: {output_dir}")
    print()


if __name__ == '__main__':
    # 스크립트 직접 실행 시 검증 및 시각화 수행
    validate_dataset()
    visualize_samples()
