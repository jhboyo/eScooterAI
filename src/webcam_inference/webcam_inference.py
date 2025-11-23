"""
Real-time PPE Detection using Webcam

Detect helmet, head, and vest in real-time using laptop camera or external webcam.

Usage:
    # Basic usage (laptop camera)
    uv run python src/webcam_inference/webcam_inference.py

    # External webcam
    uv run python src/webcam_inference/webcam_inference.py --camera 1

    # Adjust confidence
    uv run python src/webcam_inference/webcam_inference.py --conf 0.3

    # Custom resolution
    uv run python src/webcam_inference/webcam_inference.py --width 1280 --height 720

Keyboard Controls:
    Q - Quit
    S - Save Screenshot
    P - Pause/Resume
    + - Increase Confidence
    - - Decrease Confidence
    H - Toggle Help
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from utils import (
    FPSCounter,
    calculate_statistics_from_results,
    draw_statistics_overlay,
    draw_help_overlay,
    save_screenshot,
    get_available_cameras,
    initialize_camera
)


# ============================================================================
# 메인 실시간 추론 함수
# Main Real-time Inference Function
# ============================================================================

def run_realtime_inference(
    camera_id: int = 0,
    model_path: Path = None,
    conf_threshold: float = 0.25,
    width: int = None,
    height: int = None,
    output_dir: Path = None
):
    """
    실시간 PPE 탐지 수행

    Args:
        camera_id: 카메라 인덱스 (0: 노트북 내장, 1: 외부 웹캠)
        model_path: YOLO 모델 경로
        conf_threshold: 신뢰도 임계값
        width: 해상도 너비
        height: 해상도 높이
        output_dir: 스크린샷 저장 디렉토리
    """
    print("="*80)
    print("PPE Detection - Real-time Webcam Inference")
    print("="*80)
    print(f"Camera: {camera_id}")
    print(f"Model: {model_path}")
    print(f"Confidence Threshold: {conf_threshold}")
    if width and height:
        print(f"Resolution: {width}x{height}")
    print("="*80)
    print()

    # 사용 가능한 카메라 확인
    available_cameras = get_available_cameras()
    print(f"Available cameras: {available_cameras}")

    if camera_id not in available_cameras:
        print(f"Error: Camera {camera_id} is not available.")
        print(f"Please use one of: {available_cameras}")
        return

    # 모델 로드
    print("\nLoading YOLO model...")
    try:
        model = YOLO(str(model_path))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 카메라 초기화
    print(f"\nInitializing camera {camera_id}...")
    try:
        cap = initialize_camera(camera_id, width, height)
        print("Camera initialized successfully!")
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return

    # 실제 해상도 확인
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")

    # FPS 카운터 초기화
    fps_counter = FPSCounter(window_size=30)

    # 상태 변수
    paused = False
    show_help = False
    current_conf = conf_threshold

    print("\n" + "="*80)
    print("Starting real-time inference...")
    print("Press 'H' for keyboard controls")
    print("="*80 + "\n")

    # 메인 루프
    frame_count = 0
    try:
        while True:
            # 일시정지 상태가 아닐 때만 프레임 읽기
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame from camera")
                    break

                frame_count += 1

                # YOLO 추론
                results = model(frame, conf=current_conf, verbose=False)

                # 결과 시각화 (바운딩 박스)
                annotated_frame = results[0].plot()

                # 통계 계산
                stats = calculate_statistics_from_results(results)

                # FPS 업데이트
                fps = fps_counter.update()

                # 통계 오버레이 추가
                display_frame = draw_statistics_overlay(
                    annotated_frame, stats, fps, current_conf
                )
            else:
                # 일시정지 상태에서는 기존 프레임 사용
                display_frame = annotated_frame.copy()

                # 일시정지 메시지 표시
                height, width = display_frame.shape[:2]
                cv2.putText(
                    display_frame, "PAUSED (Press P to resume)",
                    (width // 2 - 200, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255), 2
                )

            # 도움말 오버레이 (H 키 누름)
            if show_help:
                display_frame = draw_help_overlay(display_frame)

            # 화면 표시
            cv2.imshow('PPE Detection - Real-time', display_frame)

            # 키보드 입력 처리
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                # 종료
                print("\nQuitting...")
                break

            elif key == ord('s') or key == ord('S'):
                # 스크린샷 저장
                filepath = save_screenshot(display_frame, str(output_dir))
                print(f"Screenshot saved: {filepath}")

            elif key == ord('p') or key == ord('P'):
                # 일시정지/재개
                paused = not paused
                status = "paused" if paused else "resumed"
                print(f"Video {status}")

            elif key == ord('h') or key == ord('H'):
                # 도움말 토글
                show_help = not show_help

            elif key == ord('+') or key == ord('='):
                # 신뢰도 증가
                current_conf = min(current_conf + 0.05, 0.95)
                print(f"Confidence threshold: {current_conf:.2f}")

            elif key == ord('-') or key == ord('_'):
                # 신뢰도 감소
                current_conf = max(current_conf - 0.05, 0.05)
                print(f"Confidence threshold: {current_conf:.2f}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # 리소스 해제
        print("\nReleasing resources...")
        cap.release()
        cv2.destroyAllWindows()

        # 최종 통계 출력
        print("\n" + "="*80)
        print("Session Summary")
        print("="*80)
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {fps_counter.get_fps():.1f}")
        print("="*80)
        print("\nThank you for using PPE Detection System!")


# ============================================================================
# 커맨드라인 인터페이스
# Command-Line Interface
# ============================================================================

def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Real-time PPE Detection using Webcam',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Laptop camera
  python webcam_inference.py

  # External webcam
  python webcam_inference.py --camera 1

  # Adjust confidence
  python webcam_inference.py --conf 0.3

  # Custom resolution
  python webcam_inference.py --width 1280 --height 720

Keyboard Controls:
  Q - Quit
  S - Save Screenshot
  P - Pause/Resume
  + - Increase Confidence
  - - Decrease Confidence
  H - Toggle Help
        """
    )

    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera index (0: laptop, 1: external webcam)'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Model file path (default: models/ppe_detection/weights/best.pt)'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )

    parser.add_argument(
        '--width', '-w',
        type=int,
        default=None,
        help='Camera width resolution'
    )

    parser.add_argument(
        '--height', '-ht',
        type=int,
        default=None,
        help='Camera height resolution'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Screenshot output directory (default: output/webcam_screenshots)'
    )

    return parser.parse_args()


# ============================================================================
# 메인 함수
# Main Function
# ============================================================================

def main():
    """메인 실행 함수"""
    args = parse_args()

    # 프로젝트 기본 디렉토리 (src/webcam_inference/webcam_inference.py)
    base_dir = Path(__file__).parent.parent.parent

    # 모델 경로 설정
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = base_dir / model_path
    else:
        model_path = base_dir / 'models' / 'ppe_detection' / 'weights' / 'best.pt'

    # 출력 디렉토리 설정
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir
    else:
        output_dir = base_dir / 'output' / 'webcam_screenshots'

    # 모델 파일 존재 확인
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("\nPlease ensure the model file exists or specify the correct path using --model")
        return

    # 실시간 추론 실행
    try:
        run_realtime_inference(
            camera_id=args.camera,
            model_path=model_path,
            conf_threshold=args.conf,
            width=args.width,
            height=args.height,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
