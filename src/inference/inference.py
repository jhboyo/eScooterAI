"""
Integrated PPE Detection Inference System

YOLOv8 model for simultaneous detection of helmet, head, and vest objects.

Features:
1. Process single image or entire directory
2. Simultaneous detection of helmet, head, vest
3. Visualization and result saving
4. Statistical analysis and safety level evaluation
5. Command-line argument support

Usage:
    # Single image inference
    uv run python src/inference/inference.py --input path/to/image.jpg

    # Directory inference
    uv run python src/inference/inference.py --input path/to/directory

    # Adjust confidence threshold
    uv run python src/inference/inference.py --input path/to/image.jpg --conf 0.3

    # Use custom model
    uv run python src/inference/inference.py --input path/to/image.jpg --model path/to/model.pt
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple
import json
from datetime import datetime

# ============================================================================
# 클래스 정의 및 설정
# Class Definitions and Settings
# ============================================================================

# 탐지 클래스 정보
# Detection class information
CLASS_NAMES = {
    0: 'helmet',  # 헬멧 착용 (안전) / Helmet worn (safe)
    1: 'head',    # 헬멧 미착용 (위험!) / No helmet (danger!)
    2: 'vest'     # 안전조끼 착용 (안전장비) / Safety vest (safety equipment)
}

# 클래스별 시각화 색상 (RGB)
# Visualization colors per class (RGB)
CLASS_COLORS = {
    0: (0, 0, 255),     # helmet - 파란색 (안전) / blue (safe)
    1: (255, 0, 0),     # head - 빨간색 (위험!) / red (danger!)
    2: (255, 255, 0)    # vest - 노란색 (안전장비) / yellow (safety equipment)
}

# ============================================================================
# 추론 및 시각화 함수
# Inference and Visualization Functions
# ============================================================================

def detect_objects(model: YOLO, image_path: Path, conf_threshold: float = 0.25) -> Tuple[np.ndarray, List[Dict]]:
    """
    이미지에서 객체 탐지 수행
    Perform object detection on image

    Args:
        model: YOLOv8 모델 / YOLOv8 model
        image_path: 이미지 파일 경로 / Image file path
        conf_threshold: 신뢰도 임계값 (기본값: 0.25) / Confidence threshold (default: 0.25)

    Returns:
        Tuple[np.ndarray, List[Dict]]: 원본 이미지(RGB) 및 탐지 결과 / Original image (RGB) and detection results
    """
    # 이미지 읽기 / Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # BGR을 RGB로 변환 (OpenCV는 BGR, matplotlib는 RGB 사용)
    # BGR to RGB conversion (OpenCV uses BGR, matplotlib uses RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 객체 탐지 수행 / Perform object detection
    results = model(image, conf=conf_threshold)

    # 탐지 결과 파싱 / Parse detection results
    detections = []
    for r in results:
        boxes = r.boxes  # 바운딩 박스 정보 / Bounding box information
        if boxes is not None:
            for box in boxes:
                # 바운딩 박스 좌표 추출 (x1, y1, x2, y2)
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # 신뢰도 점수 / Confidence score
                conf = float(box.conf[0].cpu().numpy())
                # 클래스 ID / Class ID
                cls = int(box.cls[0].cpu().numpy())

                # 탐지 결과를 딕셔너리로 저장
                # Store detection result as dictionary
                detections.append({
                    'class_id': cls,
                    'class_name': CLASS_NAMES.get(cls, f'class_{cls}'),
                    'confidence': conf,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })

    return image_rgb, detections


def visualize_detections(image_rgb: np.ndarray,
                         detections: List[Dict],
                         image_name: str,
                         output_path: Path) -> None:
    """
    탐지 결과 시각화 및 저장
    Visualize and save detection results

    Args:
        image_rgb: 원본 이미지 (RGB) / Original image (RGB)
        detections: 탐지 결과 리스트 / Detection results list
        image_name: 이미지 파일명 / Image file name
        output_path: 저장 파일 경로 / Save file path
    """
    # Figure 생성 (원본 이미지와 탐지 결과를 나란히 표시)
    # Create figure (display original and detection results side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 원본 이미지 표시 / Display original image
    ax1.imshow(image_rgb)
    ax1.set_title(f'Original: {image_name}', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 탐지 결과 이미지 표시 / Display detection results
    ax2.imshow(image_rgb)
    ax2.set_title('Detection Results', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 바운딩 박스와 레이블 그리기 / Draw bounding boxes and labels
    for det in detections:
        x1, y1, x2, y2 = det['bbox']  # 바운딩 박스 좌표 / Bounding box coordinates
        cls = det['class_id']  # 클래스 ID / Class ID
        conf = det['confidence']  # 신뢰도 / Confidence
        class_name = det['class_name']  # 클래스 이름 / Class name

        # 클래스별 색상 가져오기 / Get color by class
        color = CLASS_COLORS.get(cls, (128, 128, 128))
        color_normalized = np.array(color) / 255  # matplotlib는 0-1 범위 사용

        # 바운딩 박스 그리기 / Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,  # (왼쪽 상단 좌표, 너비, 높이)
            linewidth=2,
            edgecolor=color_normalized,
            facecolor='none'  # 투명한 내부
        )
        ax2.add_patch(rect)

        # 레이블 텍스트 추가 (클래스명: 신뢰도)
        # Add label text (class_name: confidence)
        label = f'{class_name}: {conf:.2f}'
        ax2.text(
            x1, y1 - 5,  # 바운딩 박스 위쪽에 배치
            label,
            color=color_normalized,
            fontsize=10,
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                alpha=0.7  # 반투명 배경
            )
        )

    # 통계 정보 계산 / Calculate statistics
    helmet_count = sum(1 for d in detections if d['class_name'] == 'helmet')
    head_count = sum(1 for d in detections if d['class_name'] == 'head')
    vest_count = sum(1 for d in detections if d['class_name'] == 'vest')

    # 통계 정보를 제목으로 표시 / Display statistics as title
    stats_text = f'Helmet: {helmet_count} | Head: {head_count} | Vest: {vest_count}'
    fig.suptitle(stats_text, fontsize=14, fontweight='bold')

    # 이미지 저장 / Save image
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()  # 메모리 해제 / Release memory


def calculate_statistics(all_detections: List[Dict]) -> Dict:
    """
    전체 탐지 결과에 대한 통계 계산
    Calculate statistics for all detection results

    Args:
        all_detections: 모든 이미지의 탐지 결과 / Detection results from all images

    Returns:
        Dict: 통계 정보 / Statistical information
    """
    # 각 클래스별 탐지 개수 초기화 / Initialize detection counts
    total_helmet = 0
    total_head = 0
    total_vest = 0
    total_images = len(all_detections)

    # 모든 이미지의 탐지 결과를 순회하며 개수 집계
    # Iterate through all detection results and count
    for result in all_detections:
        detections = result['detections']
        total_helmet += sum(1 for d in detections if d['class_name'] == 'helmet')
        total_head += sum(1 for d in detections if d['class_name'] == 'head')
        total_vest += sum(1 for d in detections if d['class_name'] == 'vest')

    # 헬멧 착용률 계산 / Calculate helmet wearing rate
    total_workers = total_helmet + total_head  # 전체 작업자 수 (헬멧 착용 + 미착용)
    helmet_rate = (total_helmet / total_workers * 100) if total_workers > 0 else 0

    # 안전도 평가 / Evaluate safety level
    # 90% 이상: 우수 / 70% 이상: 주의 필요 / 70% 미만: 위험
    if helmet_rate >= 90:
        safety_level = "Excellent"
    elif helmet_rate >= 70:
        safety_level = "Caution Required"
    else:
        safety_level = "Dangerous"

    return {
        'total_images': total_images,
        'total_helmet': total_helmet,
        'total_head': total_head,
        'total_vest': total_vest,
        'total_workers': total_workers,
        'helmet_rate': helmet_rate,
        'safety_level': safety_level
    }


def save_results_json(all_detections: List[Dict], statistics: Dict, output_path: Path) -> None:
    """
    결과를 JSON 파일로 저장
    Save results to JSON file

    Args:
        all_detections: 모든 탐지 결과 / All detection results
        statistics: 통계 정보 / Statistical information
        output_path: JSON 저장 경로 / JSON save path
    """
    # 결과 데이터 구조화 / Structure result data
    results = {
        'timestamp': datetime.now().isoformat(),  # 생성 시간 / Generation time
        'statistics': statistics,  # 통계 정보 / Statistical info
        'detections': all_detections  # 개별 탐지 결과 / Individual detection results
    }

    # UTF-8 인코딩으로 JSON 파일 저장 / Save JSON file with UTF-8 encoding
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ============================================================================
# 메인 추론 함수
# Main Inference Function
# ============================================================================

def run_inference(input_path: Path,
                  model_path: Path,
                  output_dir: Path,
                  conf_threshold: float = 0.25,
                  save_json: bool = True) -> None:
    """
    통합 추론 실행
    Run integrated inference

    Args:
        input_path: 입력 이미지 또는 디렉토리 경로 / Input image or directory path
        model_path: 모델 파일 경로 / Model file path
        output_dir: 출력 디렉토리 경로 / Output directory path
        conf_threshold: 신뢰도 임계값 / Confidence threshold
        save_json: JSON 결과 저장 여부 / Whether to save JSON results
    """
    # 시작 메시지 출력 / Print start message
    print("="*80)
    print("PPE Detection Integrated Inference System")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Confidence Threshold: {conf_threshold}")
    print("="*80)
    print()

    # 출력 디렉토리 생성 / Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드 / Load model
    print("Loading model...")
    model = YOLO(str(model_path))
    print("Model loaded successfully\n")

    # 입력 파일 리스트 생성 / Generate input file list
    if input_path.is_file():
        # 단일 이미지 파일인 경우 / Single image file
        image_files = [input_path]
    elif input_path.is_dir():
        # 디렉토리인 경우, 지원되는 이미지 확장자 검색
        # If directory, search for supported image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(ext))
        image_files = sorted(image_files)  # 파일명 순서대로 정렬
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    # 이미지 파일이 없으면 에러 / Error if no images found
    if not image_files:
        raise ValueError(f"No images found to process: {input_path}")

    print(f"Processing {len(image_files)} images")
    print("-"*80)

    # 각 이미지 처리 / Process each image
    all_detections = []  # 모든 탐지 결과 저장 리스트

    for idx, img_path in enumerate(image_files, 1):
        try:
            print(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")

            # 객체 탐지 수행 / Perform object detection
            image_rgb, detections = detect_objects(model, img_path, conf_threshold)

            # 결과 시각화 및 저장 / Visualize and save results
            output_path = output_dir / f'detection_{img_path.stem}.png'
            visualize_detections(image_rgb, detections, img_path.name, output_path)

            # 탐지 결과 저장 / Save detection results
            all_detections.append({
                'image_name': img_path.name,
                'image_path': str(img_path),
                'detections': detections
            })

            # 이미지별 통계 출력 / Output statistics per image
            helmet_count = sum(1 for d in detections if d['class_name'] == 'helmet')
            head_count = sum(1 for d in detections if d['class_name'] == 'head')
            vest_count = sum(1 for d in detections if d['class_name'] == 'vest')

            print(f"   Done - Helmet: {helmet_count}, Head: {head_count}, Vest: {vest_count}")

        except Exception as e:
            # 에러 발생 시 해당 이미지 건너뛰기 / Skip image on error
            print(f"   Error: {e}")
            continue

    print("-"*80)
    print()

    # 전체 통계 계산 및 출력 / Calculate and display overall statistics
    if all_detections:
        print("Analyzing overall statistics...")
        statistics = calculate_statistics(all_detections)

        print("="*80)
        print("Final Results")
        print("="*80)
        print(f"Processed Images: {statistics['total_images']}")
        print(f"Total Helmets: {statistics['total_helmet']}")
        print(f"Total Heads (No Helmet): {statistics['total_head']}")
        print(f"Total Vests: {statistics['total_vest']}")
        print(f"Total Workers: {statistics['total_workers']}")

        # 작업자가 있는 경우 착용률 및 안전도 출력
        # Display wearing rate and safety level if workers exist
        if statistics['total_workers'] > 0:
            print(f"Helmet Wearing Rate: {statistics['helmet_rate']:.1f}%")
            print(f"Safety Level: {statistics['safety_level']}")

        print("="*80)
        print(f"\nResults saved: {output_dir}")

        # JSON 파일로 결과 저장 / Save results to JSON file
        if save_json:
            json_path = output_dir / 'results.json'
            save_results_json(all_detections, statistics, json_path)
            print(f"JSON saved: {json_path}")
    else:
        print("No images processed.")


# ============================================================================
# 커맨드라인 인터페이스
# Command-Line Interface
# ============================================================================

def parse_args():
    """
    커맨드라인 인자 파싱
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='PPE Detection Integrated Inference System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 단일 이미지 추론 / Single image inference
  python inference.py --input test.jpg

  # 디렉토리 추론 / Directory inference
  python inference.py --input ./images/

  # 신뢰도 조정 / Adjust confidence
  python inference.py --input test.jpg --conf 0.3

  # 커스텀 모델 사용 / Use custom model
  python inference.py --input test.jpg --model custom_model.pt
        """
    )

    # 입력 경로 (이미지 파일 또는 디렉토리)
    # Input path (image file or directory)
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=False,
        help='Input image or directory path'
    )

    # 모델 파일 경로
    # Model file path
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=False,
        help='Model file path (default: models/ppe_detection/weights/best.pt)'
    )

    # 출력 디렉토리 경로
    # Output directory path
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=False,
        help='Output directory path (default: output/inference)'
    )

    # 신뢰도 임계값
    # Confidence threshold
    parser.add_argument(
        '--conf', '-c',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )

    # JSON 저장 비활성화 옵션
    # Disable JSON save option
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Do not save JSON results'
    )

    return parser.parse_args()


# ============================================================================
# 메인 함수
# Main Function
# ============================================================================

def main():
    """
    메인 실행 함수
    Main execution function
    """
    # 커맨드라인 인자 파싱 / Parse command-line arguments
    args = parse_args()

    # 프로젝트 기본 디렉토리 설정 / Set project base directory
    # 이 파일이 src/inference/inference.py에 있으므로 3단계 상위가 프로젝트 루트
    base_dir = Path(__file__).parent.parent.parent

    # 입력 경로 설정 / Set input path
    if args.input:
        input_path = Path(args.input)
        # 상대 경로인 경우 절대 경로로 변환 / Convert to absolute path if relative
        if not input_path.is_absolute():
            input_path = base_dir / input_path
    else:
        # 기본값: 테스트 데이터셋 / Default: test dataset
        input_path = base_dir / 'dataset' / 'data' / 'test' / 'images'

    # 모델 경로 설정 / Set model path
    if args.model:
        model_path = Path(args.model)
        # 상대 경로인 경우 절대 경로로 변환 / Convert to absolute path if relative
        if not model_path.is_absolute():
            model_path = base_dir / model_path
    else:
        # 기본값: best.pt 모델 / Default: best.pt model
        model_path = base_dir / 'models' / 'ppe_detection' / 'weights' / 'best.pt'

    # 출력 경로 설정 / Set output path
    if args.output:
        output_dir = Path(args.output)
        # 상대 경로인 경우 절대 경로로 변환 / Convert to absolute path if relative
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir
    else:
        # 기본값: output/inference 디렉토리 / Default: output/inference directory
        output_dir = base_dir / 'output' / 'inference'

    # 경로 유효성 검증 / Validate paths
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}")
        return

    if not model_path.exists():
        print(f"Model file does not exist: {model_path}")
        return

    # 추론 실행 / Run inference
    try:
        run_inference(
            input_path=input_path,
            model_path=model_path,
            output_dir=output_dir,
            conf_threshold=args.conf,
            save_json=not args.no_json  # --no-json 플래그가 있으면 저장하지 않음
        )
    except Exception as e:
        # 에러 발생 시 상세 에러 메시지 출력 / Print detailed error message on exception
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
