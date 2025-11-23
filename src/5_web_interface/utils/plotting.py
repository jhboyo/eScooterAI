"""
바운딩 박스 그리기 및 결과 시각화 유틸리티

PPE 탐지 결과를 이미지에 시각화하는 기능을 제공합니다.

주요 기능:
- 바운딩 박스 그리기
- 클래스별 색상 구분
- 신뢰도 라벨 표시
- 원본/결과 비교 뷰

Author: Safety Vision AI Team
Date: 2025-11-22
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
import io


# ============================================================================
# 클래스별 색상 정의 (RGB)
# ============================================================================

CLASS_COLORS = {
    'helmet': (0, 128, 255),      # 파란색 (안전)
    'head': (255, 0, 0),          # 빨간색 (위험)
    'vest': (255, 200, 0)         # 노란색/주황색 (주의)
}

# 기본 색상 (알 수 없는 클래스용)
DEFAULT_COLOR = (128, 128, 128)  # 회색


# ============================================================================
# 폰트 설정
# ============================================================================

def get_font(size: int) -> ImageFont.FreeTypeFont:
    """
    적절한 폰트 가져오기

    Args:
        size: 폰트 크기

    Returns:
        ImageFont: 폰트 객체
    """
    try:
        # macOS 기본 폰트
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except:
        try:
            # Windows 기본 폰트
            return ImageFont.truetype("arial.ttf", size)
        except:
            # 기본 폰트
            return ImageFont.load_default()


def get_font_size(image_size: Tuple[int, int]) -> int:
    """
    이미지 크기에 따른 적절한 폰트 크기 계산

    Args:
        image_size: (width, height)

    Returns:
        int: 폰트 크기
    """
    width, height = image_size
    # 이미지 크기에 비례하여 폰트 크기 결정
    base_size = min(width, height)
    font_size = max(12, int(base_size / 40))
    return min(font_size, 32)  # 최대 32


# ============================================================================
# 바운딩 박스 그리기
# ============================================================================

def draw_bounding_box(
    draw: ImageDraw.Draw,
    bbox: List[float],
    class_name: str,
    confidence: float,
    image_size: Tuple[int, int]
):
    """
    단일 바운딩 박스 그리기

    Args:
        draw: PIL ImageDraw 객체
        bbox: [x1, y1, x2, y2] 좌표
        class_name: 클래스명 (helmet, head, vest)
        confidence: 신뢰도 (0.0 ~ 1.0)
        image_size: 이미지 크기 (width, height)
    """
    x1, y1, x2, y2 = bbox

    # 클래스별 색상 가져오기
    color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

    # 선 두께 (이미지 크기에 비례)
    line_width = max(2, int(min(image_size) / 200))

    # 바운딩 박스 그리기
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

    # 라벨 텍스트 생성
    label = f"{class_name}: {confidence:.2f}"

    # 폰트 크기 계산
    font_size = get_font_size(image_size)
    font = get_font(font_size)

    # 텍스트 크기 계산
    try:
        # PIL 최신 버전
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
    except AttributeError:
        # PIL 구버전 호환
        text_width, text_height = draw.textsize(label, font=font)

    # 라벨 배경 위치 (박스 상단)
    padding = 4
    label_x1 = x1
    label_y1 = max(0, y1 - text_height - padding * 2)
    label_x2 = x1 + text_width + padding * 2
    label_y2 = y1

    # 라벨 배경 그리기
    draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill=color)

    # 라벨 텍스트 그리기 (흰색)
    draw.text(
        (label_x1 + padding, label_y1 + padding),
        label,
        fill=(255, 255, 255),
        font=font
    )


# ============================================================================
# 이미지에 탐지 결과 시각화
# ============================================================================

def visualize_detections(
    image: Image.Image,
    detections: List[Dict]
) -> Image.Image:
    """
    이미지에 탐지 결과 시각화

    Args:
        image: 원본 PIL Image
        detections: 탐지 결과 리스트
            각 항목: {'bbox': [x1,y1,x2,y2], 'class_name': str, 'confidence': float}

    Returns:
        Image: 바운딩 박스가 그려진 이미지
    """
    # 원본 이미지 복사 (원본 유지)
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    # 이미지 크기
    image_size = image.size
    width, height = image_size

    # 각 탐지 결과에 대해 바운딩 박스 그리기
    for detection in detections:
        draw_bounding_box(
            draw=draw,
            bbox=detection['bbox'],
            class_name=detection['class_name'],
            confidence=detection['confidence'],
            image_size=image_size
        )

    # 클래스별 카운트 계산
    class_counts = {}
    for detection in detections:
        cls = detection['class_name']
        class_counts[cls] = class_counts.get(cls, 0) + 1

    # 탐지 정보 텍스트 생성
    helmet_count = class_counts.get('helmet', 0)
    head_count = class_counts.get('head', 0)
    vest_count = class_counts.get('vest', 0)
    person_count = helmet_count + head_count  # Person = Helmet + Head
    detection_text = f"Detections: Person={person_count}, Helmet={helmet_count}, Head={head_count}, Vest={vest_count}"

    # 텍스트 폰트 크기 (이미지 크기에 비례, 더 작게)
    text_font_size = max(12, int(min(width, height) / 50))
    text_font = get_font(text_font_size)

    # 텍스트 크기 계산
    try:
        bbox_text = draw.textbbox((0, 0), detection_text, font=text_font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
    except AttributeError:
        text_width, text_height = draw.textsize(detection_text, font=text_font)

    # 텍스트 배경 위치 (상단 중앙)
    padding = 10
    bg_x1 = (width - text_width) // 2 - padding
    bg_y1 = padding
    bg_x2 = (width + text_width) // 2 + padding
    bg_y2 = padding + text_height + padding * 2

    # 배경 그리기 (흰색, 약간 투명)
    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(255, 255, 255, 230))

    # 테두리 그리기
    draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], outline=(0, 0, 0), width=2)

    # 텍스트 그리기 (검은색, 굵게)
    text_x = (width - text_width) // 2
    text_y = padding + padding
    draw.text((text_x, text_y), detection_text, fill=(0, 0, 0), font=text_font)

    return result_image


# ============================================================================
# 여러 이미지 일괄 시각화
# ============================================================================

def visualize_batch(
    images: List[Image.Image],
    results: List[Dict]
) -> List[Image.Image]:
    """
    여러 이미지에 대한 일괄 시각화

    Args:
        images: 원본 PIL Image 리스트
        results: 추론 결과 리스트

    Returns:
        List[Image]: 시각화된 이미지 리스트
    """
    visualized_images = []

    for image, result in zip(images, results):
        visualized = visualize_detections(image, result['detections'])
        visualized_images.append(visualized)

    return visualized_images


# ============================================================================
# Streamlit 컴포넌트: 원본/결과 비교 뷰
# ============================================================================

def render_comparison_view(
    original_images: List[Image.Image],
    results: List[Dict],
    uploaded_files: List
):
    """
    원본/결과 비교 뷰 렌더링 (모든 이미지를 세로로 나열)

    Args:
        original_images: 원본 이미지 리스트
        results: 추론 결과 리스트
        uploaded_files: 업로드된 파일 정보 리스트
    """
    if not results or not original_images:
        st.warning("표시할 결과가 없습니다.")
        return

    st.markdown("---")
    st.subheader("🖼️ 탐지 결과 비교")
    st.caption("각 이미지의 원본과 탐지 결과를 비교합니다")

    total_images = len(original_images)

    # 세션 상태에 결과 이미지 저장 (다운로드용)
    if 'result_images' not in st.session_state:
        st.session_state.result_images = []

    # 모든 이미지를 순회하며 표시 (2열 레이아웃)
    for idx, (original, result) in enumerate(zip(original_images, results)):
        # 이미지 구분선
        if idx > 0:
            st.markdown("---")

        # 이미지 헤더 (폰트 크기 증가)
        filename = uploaded_files[idx].name if idx < len(uploaded_files) else f"이미지 {idx+1}"
        num_detections = result['num_detections']

        st.markdown(f"## 📷 {idx+1}. {filename}")
        st.caption(f"🔍 {num_detections}개 객체 탐지됨 | 📐 {original.size[0]} × {original.size[1]}")

        # 결과 이미지 생성
        result_image = visualize_detections(original, result['detections'])

        # 세션 상태에 저장
        if len(st.session_state.result_images) <= idx:
            st.session_state.result_images.append(result_image)
        else:
            st.session_state.result_images[idx] = result_image

        # 2열 레이아웃으로 원본/결과 비교
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**원본**")
            st.image(original, width='stretch')

        with col2:
            st.markdown("**탐지 결과**")
            st.image(result_image, width='stretch')

        # 탐지 정보가 있으면 표시
        if num_detections > 0:
            # 클래스별 통계
            class_counts = {}
            for det in result['detections']:
                cls = det['class_name']
                class_counts[cls] = class_counts.get(cls, 0) + 1

            helmet_count = class_counts.get('helmet', 0)
            head_count = class_counts.get('head', 0)
            vest_count = class_counts.get('vest', 0)
            person_count = helmet_count + head_count

            # 간단한 메트릭 표시
            stat_cols = st.columns(4)

            with stat_cols[0]:
                st.metric("👷 Person", f"{person_count}명")

            with stat_cols[1]:
                st.metric("🔵 Helmet", f"{helmet_count}개")

            with stat_cols[2]:
                if head_count > 0:
                    st.metric("🔴 Head", f"{head_count}개", delta="⚠️ 미착용", delta_color="inverse")
                else:
                    st.metric("🔴 Head", "0개")

            with stat_cols[3]:
                st.metric("🟡 Vest", f"{vest_count}개")

            # 헬멧 착용률 계산 및 표시
            if person_count > 0:
                helmet_rate = (helmet_count / person_count) * 100

                # 안전 수준을 한 줄로 표시
                if helmet_rate >= 90:
                    st.success(f"✅ **Excellent** - 헬멧 착용률 {helmet_rate:.1f}% (매우 안전)")
                elif helmet_rate >= 70:
                    st.warning(f"⚠️ **Caution** - 헬멧 착용률 {helmet_rate:.1f}% (주의 필요)")
                else:
                    st.error(f"🚨 **Dangerous** - 헬멧 착용률 {helmet_rate:.1f}% (위험!)")
        else:
            st.info("이 이미지에서는 PPE(개인보호구)가 탐지되지 않았습니다.")


# ============================================================================
# 이미지를 바이트로 변환 (다운로드용)
# ============================================================================

def image_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    """
    PIL Image를 바이트로 변환

    Args:
        image: PIL Image
        format: 이미지 포맷 (PNG, JPEG 등)

    Returns:
        bytes: 이미지 바이트 데이터
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.getvalue()
