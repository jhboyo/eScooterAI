"""
YOLOv8 모델 추론 유틸리티
Streamlit 웹 인터페이스를 위한 PPE 탐지 추론 기능

Author: Safety Vision AI Team
Date: 2025-11-22
"""

import streamlit as st
from ultralytics import YOLO
from pathlib import Path
import time
from typing import List, Dict, Optional
import numpy as np
from PIL import Image


# ============================================================================
# 모델 로드 (캐싱)
# ============================================================================

@st.cache_resource
def load_model(model_path: str) -> Optional[YOLO]:
    """
    YOLOv8 모델 로드 (세션 간 캐싱)

    Args:
        model_path: 모델 파일 경로 (예: "models/ppe_detection/weights/best.pt")

    Returns:
        YOLO: 로드된 YOLOv8 모델 객체
        None: 모델 로드 실패 시

    Note:
        @st.cache_resource 데코레이터를 사용하여 모델을 한 번만 로드하고
        세션 간에 공유합니다. 이는 성능 향상에 중요합니다.
    """
    try:
        # 모델 파일 경로 확인
        model_file = Path(model_path)

        if not model_file.exists():
            st.error(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")

            # 디버깅 정보
            import os
            with st.expander("🔍 디버깅 정보"):
                st.code(f"""
현재 파일: {Path(__file__).resolve()}
모델 경로: {model_file}
파일 존재: {model_file.exists()}
작업 디렉토리: {os.getcwd()}
디렉토리 내용:
{chr(10).join([f"  - {p}" for p in Path('.').glob('**/*') if p.is_file()][:20])}
                """)

            st.info("💡 프로젝트 루트에서 다음 명령으로 학습을 먼저 수행하세요:\n```bash\nuv run python src/2_training/train.py\n```")
            return None

        # 모델 로드 (스피너 표시)
        with st.spinner(f"🔄 YOLOv8 모델 로딩 중... ({model_file.name})"):
            model = YOLO(str(model_file))

        st.success(f"✅ 모델 로드 완료: {model_file.name}")

        # 모델 정보 출력 (디버그용)
        class_list = ', '.join([f"{k}: {v}" for k, v in model.names.items()])
        st.sidebar.info(f"""
        **모델 정보**
        - 파일: {model_file.name}
        - 클래스 수: {len(model.names)}개
        - 클래스 목록: {class_list}
        - 장치: {'GPU (CUDA)' if model.device.type == 'cuda' else 'CPU'}
        """)

        # 클래스 확인용 경고 메시지
        expected_classes = {'helmet', 'head', 'vest'}
        actual_classes = set(model.names.values())
        if not expected_classes.issubset(actual_classes):
            missing = expected_classes - actual_classes
            st.sidebar.warning(f"⚠️ 예상 클래스 누락: {missing}")

        return model

    except Exception as e:
        st.error(f"❌ 모델 로드 중 오류 발생: {str(e)}")
        st.exception(e)
        return None


# ============================================================================
# 단일 이미지 추론
# ============================================================================

def run_inference_single(
    model: YOLO,
    image: Image.Image,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
    debug: bool = False
) -> Dict:
    """
    단일 이미지에 대한 PPE 탐지 추론

    Args:
        model: YOLOv8 모델 객체
        image: PIL Image 객체
        conf: 신뢰도 임계값 (0.0 ~ 1.0)
        iou: IoU 임계값 (Non-Maximum Suppression용)
        max_det: 최대 탐지 객체 수
        debug: 디버그 정보 출력 여부

    Returns:
        Dict: 추론 결과
            - detections: 탐지된 객체 리스트
            - image_shape: 이미지 크기 (width, height)
            - inference_time: 추론 시간 (초)
            - num_detections: 탐지된 객체 수
            - debug_info: 디버그 정보 (debug=True일 때)
    """
    try:
        # YOLOv8 추론 실행
        results = model(
            image,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False  # 콘솔 출력 억제
        )[0]

        # 결과 파싱
        boxes = results.boxes
        detections = []
        debug_info = {
            'total_boxes': len(boxes) if boxes is not None else 0,
            'class_distribution': {},
            'all_detections': []  # 모든 탐지 결과 (필터링 전)
        }

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # 바운딩 박스 좌표 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # 신뢰도 점수
                conf_score = float(box.conf[0].cpu().numpy())

                # 클래스 ID 및 이름
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = results.names[cls_id]

                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf_score,
                    'class_id': cls_id,
                    'class_name': cls_name
                }

                detections.append(detection)

                # 디버그 정보 수집
                if debug:
                    debug_info['all_detections'].append(detection)
                    if cls_name not in debug_info['class_distribution']:
                        debug_info['class_distribution'][cls_name] = 0
                    debug_info['class_distribution'][cls_name] += 1

        # 추론 시간 (밀리초 → 초)
        inference_time = results.speed.get('inference', 0) / 1000

        result = {
            'detections': detections,
            'image_shape': image.size,  # (width, height)
            'inference_time': inference_time,
            'num_detections': len(detections)
        }

        if debug:
            result['debug_info'] = debug_info
            # 콘솔에 디버그 정보 출력
            print(f"\n=== Debug Info ===")
            print(f"Total detections: {debug_info['total_boxes']}")
            print(f"Class distribution: {debug_info['class_distribution']}")
            for i, det in enumerate(debug_info['all_detections'][:10]):  # 최대 10개만 출력
                print(f"  [{i+1}] {det['class_name']}: {det['confidence']:.3f}")

        return result

    except Exception as e:
        st.error(f"❌ 추론 중 오류 발생: {str(e)}")
        return {
            'detections': [],
            'image_shape': image.size if hasattr(image, 'size') else (0, 0),
            'inference_time': 0,
            'num_detections': 0,
            'error': str(e)
        }


# ============================================================================
# 배치 이미지 추론 (진행 상태 표시)
# ============================================================================

def run_inference_batch(
    model: YOLO,
    images: List[Image.Image],
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 300,
    show_progress: bool = True,  # 호환성을 위해 파라미터는 유지
    debug: bool = False
) -> List[Dict]:
    """
    여러 이미지에 대한 배치 추론

    Args:
        model: YOLOv8 모델 객체
        images: PIL Image 객체 리스트
        conf: 신뢰도 임계값
        iou: IoU 임계값
        max_det: 최대 탐지 객체 수
        show_progress: (사용되지 않음, 호환성 유지)
        debug: 디버그 정보 출력 여부

    Returns:
        List[Dict]: 각 이미지의 추론 결과 리스트
    """
    results = []

    # 각 이미지에 대해 추론 실행
    for idx, image in enumerate(images):
        # 단일 이미지 추론
        result = run_inference_single(model, image, conf, iou, max_det, debug=debug)
        result['image_index'] = idx
        results.append(result)

    return results


# ============================================================================
# 추론 결과 요약 통계
# ============================================================================

def summarize_results(results: List[Dict]) -> Dict:
    """
    배치 추론 결과의 요약 통계 계산

    Args:
        results: run_inference_batch()의 반환값

    Returns:
        Dict: 요약 통계
            - total_images: 총 이미지 수
            - total_detections: 총 탐지 객체 수
            - avg_detections_per_image: 이미지당 평균 탐지 수
            - class_counts: 클래스별 탐지 수
    """
    total_detections = 0
    class_counts = {}

    for result in results:
        total_detections += result['num_detections']

        for det in result['detections']:
            cls_name = det['class_name']
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    return {
        'total_images': len(results),
        'total_detections': total_detections,
        'avg_detections_per_image': total_detections / len(results) if results else 0,
        'class_counts': class_counts
    }


# ============================================================================
# 모델 경로 헬퍼 함수
# ============================================================================

def get_model_path(model_name: str) -> Path:
    """
    모델 이름으로부터 전체 경로 반환

    Args:
        model_name: 'best.pt' 또는 'last.pt'

    Returns:
        Path: 모델 파일의 전체 경로
    """
    import os

    # 현재 파일의 절대 경로
    current_file = Path(__file__).resolve()

    # utils/inference.py 위치에서 프로젝트 루트 찾기
    # 방법: models 디렉토리가 있는 곳을 찾을 때까지 상위로 이동

    # 시도 1: utils/inference.py -> utils -> project_root (HF Spaces)
    potential_root = current_file.parent.parent
    if (potential_root / "models" / "ppe_detection" / "weights" / model_name).exists():
        return potential_root / "models" / "ppe_detection" / "weights" / model_name

    # 시도 2: utils/inference.py -> utils -> 5_web_interface -> project_root (deploy/huggingface)
    potential_root = current_file.parent.parent.parent
    if (potential_root / "models" / "ppe_detection" / "weights" / model_name).exists():
        return potential_root / "models" / "ppe_detection" / "weights" / model_name

    # 시도 3: utils/inference.py -> utils -> 5_web_interface -> src -> project_root (로컬)
    potential_root = current_file.parent.parent.parent.parent
    if (potential_root / "models" / "ppe_detection" / "weights" / model_name).exists():
        return potential_root / "models" / "ppe_detection" / "weights" / model_name

    # 모든 시도 실패 - 기본 경로 반환 (에러 메시지용)
    # HF Spaces를 기본으로 가정
    return current_file.parent.parent / "models" / "ppe_detection" / "weights" / model_name
