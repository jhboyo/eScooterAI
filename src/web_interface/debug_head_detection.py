"""
Head 클래스 탐지 디버그 스크립트

모델이 head 클래스를 제대로 탐지하는지 확인하기 위한 디버그 스크립트
"""

from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import sys

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_head_detection(model_path: str, test_image_path: str = None):
    """
    Head 클래스 탐지 테스트

    Args:
        model_path: 모델 파일 경로
        test_image_path: 테스트 이미지 경로 (None이면 테스트 데이터셋에서 가져옴)
    """
    print("=" * 80)
    print("Head 클래스 탐지 디버그")
    print("=" * 80)

    # 모델 로드
    print(f"\n1. 모델 로드: {model_path}")
    model = YOLO(model_path)

    print(f"   - 클래스 매핑: {model.names}")
    print(f"   - 클래스 개수: {len(model.names)}")
    print(f"   - 장치: {model.device}")

    # 테스트 이미지 경로 설정
    if test_image_path is None:
        # 테스트 데이터셋에서 이미지 찾기
        test_images_dir = project_root / "dataset" / "data" / "test" / "images"

        if test_images_dir.exists():
            test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
            if test_images:
                test_image_path = str(test_images[0])
                print(f"\n2. 테스트 이미지: {test_image_path}")
            else:
                print("\n❌ 테스트 이미지를 찾을 수 없습니다.")
                return
        else:
            print(f"\n❌ 테스트 이미지 디렉토리를 찾을 수 없습니다: {test_images_dir}")
            return
    else:
        print(f"\n2. 테스트 이미지: {test_image_path}")

    # 이미지 로드
    image = Image.open(test_image_path)
    print(f"   - 이미지 크기: {image.size}")
    print(f"   - 이미지 모드: {image.mode}")

    # 다양한 신뢰도 임계값으로 테스트
    confidence_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    print("\n3. 다양한 신뢰도 임계값으로 탐지 테스트")
    print("-" * 80)

    for conf_threshold in confidence_thresholds:
        print(f"\n   신뢰도 임계값: {conf_threshold}")

        # 추론 실행
        results = model(image, conf=conf_threshold, verbose=False)[0]
        boxes = results.boxes

        if boxes is not None and len(boxes) > 0:
            # 클래스별 탐지 수 집계
            class_counts = {'helmet': 0, 'head': 0, 'vest': 0}
            detections = []

            for box in boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = results.names[cls_id]
                conf_score = float(box.conf[0].cpu().numpy())

                if cls_name in class_counts:
                    class_counts[cls_name] += 1

                detections.append({
                    'class': cls_name,
                    'confidence': conf_score
                })

            # 결과 출력
            print(f"   - 총 탐지 수: {len(boxes)}")
            print(f"   - Helmet: {class_counts['helmet']}")
            print(f"   - Head: {class_counts['head']} {'🔴 (미착용)' if class_counts['head'] > 0 else ''}")
            print(f"   - Vest: {class_counts['vest']}")

            # Head 클래스 상세 정보
            head_detections = [d for d in detections if d['class'] == 'head']
            if head_detections:
                print(f"\n   Head 클래스 상세:")
                for i, det in enumerate(head_detections[:5], 1):
                    print(f"      [{i}] 신뢰도: {det['confidence']:.3f}")
        else:
            print(f"   - 탐지된 객체 없음")

    print("\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)

    # 최종 분석
    print("\n📊 분석:")
    print("1. Head 클래스가 전혀 탐지되지 않는 경우:")
    print("   → 모델이 head 클래스를 학습하지 못했거나 데이터가 부족할 수 있습니다.")
    print("   → 테스트 이미지에 head 클래스가 없을 수 있습니다.")
    print("\n2. 낮은 신뢰도 임계값에서만 탐지되는 경우:")
    print("   → 모델의 head 클래스 탐지 성능이 낮습니다.")
    print("   → 추가 학습 또는 데이터 증강이 필요할 수 있습니다.")
    print("\n3. 높은 신뢰도 임계값에서도 잘 탐지되는 경우:")
    print("   → 모델이 정상적으로 작동하고 있습니다.")
    print("   → 웹 인터페이스에서 탐지되지 않는다면 코드 문제일 수 있습니다.")


if __name__ == "__main__":
    # 모델 경로
    model_path = project_root / "models" / "ppe_detection" / "weights" / "best.pt"

    if not model_path.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("먼저 모델을 학습해주세요:")
        print("uv run python src/training/train.py")
        sys.exit(1)

    # 테스트 실행
    test_head_detection(str(model_path))
