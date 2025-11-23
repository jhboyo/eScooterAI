"""
Utility Functions for Webcam Real-time Inference

FPS calculation, statistics computation, and visualization helpers.
"""

import time
import cv2
import numpy as np
from typing import List, Dict, Tuple
from collections import deque


# ============================================================================
# 클래스 정의
# Class Definitions
# ============================================================================

CLASS_NAMES = {
    0: 'helmet',  # 헬멧 착용
    1: 'head',    # 헬멧 미착용 (위험!)
    2: 'vest'     # 안전조끼
}

CLASS_COLORS_BGR = {
    0: (255, 0, 0),     # helmet - 파란색
    1: (0, 0, 255),     # head - 빨간색 (위험!)
    2: (0, 255, 255)    # vest - 노란색
}


# ============================================================================
# FPS 계산
# FPS Calculation
# ============================================================================

class FPSCounter:
    """FPS (Frames Per Second) 계산 클래스"""

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: FPS 평균 계산을 위한 윈도우 크기 (기본: 30프레임)
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = time.time()

    def update(self) -> float:
        """
        현재 프레임의 FPS 계산 및 업데이트

        Returns:
            float: 평균 FPS
        """
        current_time = time.time()
        elapsed_time = current_time - self.last_time

        if elapsed_time > 0:
            fps = 1.0 / elapsed_time
            self.frame_times.append(fps)

        self.last_time = current_time

        # 평균 FPS 반환
        if len(self.frame_times) > 0:
            return sum(self.frame_times) / len(self.frame_times)
        return 0.0

    def get_fps(self) -> float:
        """
        현재 평균 FPS 반환

        Returns:
            float: 평균 FPS
        """
        if len(self.frame_times) > 0:
            return sum(self.frame_times) / len(self.frame_times)
        return 0.0


# ============================================================================
# 통계 계산
# Statistics Calculation
# ============================================================================

def calculate_statistics_from_results(results) -> Dict:
    """
    YOLO 결과에서 통계 계산

    Args:
        results: YOLO 추론 결과

    Returns:
        Dict: 통계 정보
            - helmet_count: 헬멧 착용자 수
            - head_count: 헬멧 미착용자 수
            - vest_count: 조끼 착용자 수
            - total_workers: 전체 작업자 수
            - helmet_rate: 헬멧 착용률 (%)
            - safety_level: 안전 수준 (Excellent/Caution/Dangerous)
    """
    helmet_count = 0
    head_count = 0
    vest_count = 0

    # 탐지된 객체 순회
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0].cpu().numpy())
                if cls == 0:  # helmet
                    helmet_count += 1
                elif cls == 1:  # head
                    head_count += 1
                elif cls == 2:  # vest
                    vest_count += 1

    # 전체 작업자 수
    total_workers = helmet_count + head_count

    # 헬멧 착용률 계산
    if total_workers > 0:
        helmet_rate = (helmet_count / total_workers) * 100
    else:
        helmet_rate = 0.0

    # 안전 수준 평가
    if helmet_rate >= 90:
        safety_level = "Excellent"
        safety_color = (0, 255, 0)  # 녹색
    elif helmet_rate >= 70:
        safety_level = "Caution"
        safety_color = (0, 165, 255)  # 주황색
    else:
        safety_level = "Dangerous"
        safety_color = (0, 0, 255)  # 빨간색

    return {
        'helmet_count': helmet_count,
        'head_count': head_count,
        'vest_count': vest_count,
        'total_workers': total_workers,
        'helmet_rate': helmet_rate,
        'safety_level': safety_level,
        'safety_color': safety_color
    }


# ============================================================================
# 시각화
# Visualization
# ============================================================================

def draw_statistics_overlay(frame: np.ndarray,
                            stats: Dict,
                            fps: float,
                            conf_threshold: float) -> np.ndarray:
    """
    프레임에 통계 정보 오버레이 추가

    Args:
        frame: 원본 프레임
        stats: 통계 정보
        fps: 현재 FPS
        conf_threshold: 현재 신뢰도 임계값

    Returns:
        np.ndarray: 오버레이가 추가된 프레임
    """
    height, width = frame.shape[:2]

    # 반투명 배경 그리기 (상단)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (500, 180), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # 텍스트 시작 위치
    y_offset = 35
    line_height = 30

    # FPS 표시
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (20, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (255, 255, 255), 2
    )
    y_offset += line_height

    # 탐지 통계
    text = f"Helmet: {stats['helmet_count']} | Head: {stats['head_count']} | Vest: {stats['vest_count']}"
    cv2.putText(
        frame, text,
        (20, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2
    )
    y_offset += line_height

    # 작업자 및 착용률
    if stats['total_workers'] > 0:
        text = f"Workers: {stats['total_workers']} | Wearing Rate: {stats['helmet_rate']:.1f}%"
        cv2.putText(
            frame, text,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2
        )
        y_offset += line_height

        # 안전 수준
        text = f"Safety: {stats['safety_level']}"
        cv2.putText(
            frame, text,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            stats['safety_color'], 2
        )

    # 신뢰도 임계값 표시 (우측 상단)
    cv2.putText(
        frame, f"Conf: {conf_threshold:.2f}",
        (width - 180, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2
    )

    # 경고 메시지 (헬멧 미착용자가 있을 경우)
    if stats['head_count'] > 0:
        # 화면 중앙 상단에 경고 표시
        warning_text = f"WARNING: {stats['head_count']} Worker(s) Without Helmet!"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (width - text_size[0]) // 2

        # 경고 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (text_x - 10, height - 70), (text_x + text_size[0] + 10, height - 30), (0, 0, 255), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # 경고 텍스트
        cv2.putText(
            frame, warning_text,
            (text_x, height - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (255, 255, 255), 2
        )

    return frame


def draw_help_overlay(frame: np.ndarray) -> np.ndarray:
    """
    도움말 오버레이 표시

    Args:
        frame: 원본 프레임

    Returns:
        np.ndarray: 도움말이 추가된 프레임
    """
    height, width = frame.shape[:2]

    # 반투명 배경
    overlay = frame.copy()
    cv2.rectangle(overlay, (width // 4, height // 4), (3 * width // 4, 3 * height // 4), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

    # 도움말 텍스트
    help_lines = [
        "Keyboard Controls",
        "",
        "Q - Quit",
        "S - Save Screenshot",
        "P - Pause/Resume",
        "+ - Increase Confidence",
        "- - Decrease Confidence",
        "H - Toggle Help",
        "",
        "Press H to close"
    ]

    y_offset = height // 4 + 40
    line_height = 35

    for line in help_lines:
        if line == "Keyboard Controls":
            font_scale = 1.2
            thickness = 3
            color = (0, 255, 255)
        elif line == "":
            y_offset += line_height // 2
            continue
        else:
            font_scale = 0.8
            thickness = 2
            color = (255, 255, 255)

        cv2.putText(
            frame, line,
            (width // 4 + 30, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            color, thickness
        )
        y_offset += line_height

    return frame


def save_screenshot(frame: np.ndarray, output_dir: str) -> str:
    """
    현재 프레임을 스크린샷으로 저장

    Args:
        frame: 저장할 프레임
        output_dir: 저장 디렉토리

    Returns:
        str: 저장된 파일 경로
    """
    from pathlib import Path
    from datetime import datetime

    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 파일명 생성 (타임스탬프)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"webcam_screenshot_{timestamp}.png"
    filepath = output_path / filename

    # 이미지 저장
    cv2.imwrite(str(filepath), frame)

    return str(filepath)


# ============================================================================
# 카메라 유틸리티
# Camera Utilities
# ============================================================================

def get_available_cameras(max_cameras: int = 5) -> List[int]:
    """
    사용 가능한 카메라 인덱스 찾기

    Args:
        max_cameras: 확인할 최대 카메라 개수

    Returns:
        List[int]: 사용 가능한 카메라 인덱스 리스트
    """
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def initialize_camera(camera_id: int, width: int = None, height: int = None) -> cv2.VideoCapture:
    """
    카메라 초기화

    Args:
        camera_id: 카메라 인덱스
        width: 해상도 너비 (선택사항)
        height: 해상도 높이 (선택사항)

    Returns:
        cv2.VideoCapture: 초기화된 카메라 객체
    """
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    # 해상도 설정 (지정된 경우)
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap
