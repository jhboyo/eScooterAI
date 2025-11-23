"""
이미지 업로더 컴포넌트

미리보기 및 검증 기능을 제공하는 이미지 업로드 모듈입니다.

주요 기능:
- 다중 파일 업로드 지원
- 썸네일 미리보기
- 파일 형식 검증
- 파일 크기 검증
- 세션 상태 관리
"""

import streamlit as st
from PIL import Image
from pathlib import Path
from typing import List, Optional
import io

# 지원하는 이미지 형식
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png', 'webp', 'bmp']
MAX_FILE_SIZE_MB = 10  # 최대 파일 크기 (MB)

# ============================================================================
# 유틸리티 함수
# ============================================================================

def validate_image_file(uploaded_file) -> tuple[bool, str]:
    """
    업로드된 이미지 파일 검증

    Args:
        uploaded_file: Streamlit UploadedFile 객체

    Returns:
        tuple: (검증_성공_여부, 에러_메시지)
    """
    # 파일 확장자 확인
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in SUPPORTED_FORMATS:
        return False, f"지원하지 않는 형식: {file_extension}. 지원 형식: {', '.join(SUPPORTED_FORMATS)}"

    # 파일 크기 확인
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"파일이 너무 큽니다: {file_size_mb:.2f}MB. 최대: {MAX_FILE_SIZE_MB}MB"

    # 이미지 파일로 열기 시도
    try:
        img = Image.open(uploaded_file)
        img.verify()  # 유효한 이미지인지 검증
        uploaded_file.seek(0)  # 파일 포인터 리셋
        return True, ""
    except Exception as e:
        return False, f"유효하지 않은 이미지 파일: {str(e)}"


def get_image_info(uploaded_file) -> dict:
    """
    이미지 정보 추출

    Args:
        uploaded_file: Streamlit UploadedFile 객체

    Returns:
        dict: 이미지 정보 (크기, 해상도, 형식)
    """
    try:
        img = Image.open(uploaded_file)
        uploaded_file.seek(0)

        return {
            'filename': uploaded_file.name,
            'size_mb': uploaded_file.size / (1024 * 1024),
            'width': img.width,
            'height': img.height,
            'format': img.format,
            'mode': img.mode
        }
    except Exception as e:
        return {
            'filename': uploaded_file.name,
            'error': str(e)
        }


# ============================================================================
# 업로드 컴포넌트
# ============================================================================

def render_image_uploader(key: str = "image_uploader") -> Optional[List]:
    """
    이미지 업로더 렌더링

    Args:
        key: 업로더 위젯의 고유 키

    Returns:
        업로드된 파일 리스트 또는 None
    """
    st.markdown("### 📁 이미지 업로드")

    # 안내 문구
    st.info("📸 **JPG, PNG, WEBP, BMP** 형식 지원 (최대 10MB)")

    # 동적 키로 파일 업로더 생성 (삭제 시 위젯 리셋용)
    uploader_key = f"{key}_{st.session_state.uploader_key}"

    # CSS로 업로더 버튼 스타일 크게 만들기
    st.markdown("""
        <style>
        div[data-testid="stFileUploader"] {
            width: 100%;
        }
        div[data-testid="stFileUploader"] > label {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1f4068;
        }
        div[data-testid="stFileUploader"] > div {
            padding: 2rem;
            border: 2px dashed #1f4068;
            border-radius: 10px;
            background-color: #f8f9fa;
        }
        div[data-testid="stFileUploader"] button {
            font-size: 1.1rem;
            padding: 0.75rem 2rem;
            background-color: #1f4068;
            color: white;
            border-radius: 8px;
        }
        div[data-testid="stFileUploader"] button:hover {
            background-color: #163456;
        }
        </style>
    """, unsafe_allow_html=True)

    # 파일 업로더 (라벨 표시)
    uploaded_files = st.file_uploader(
        "🖼️ 이미지 파일을 선택하거나 드래그 앤 드롭하세요",
        type=SUPPORTED_FORMATS,
        accept_multiple_files=True,
        help=f"여러 이미지를 한 번에 업로드할 수 있습니다 (각 파일 최대 {MAX_FILE_SIZE_MB}MB)",
        key=uploader_key
    )

    if uploaded_files:
        # 모든 파일 검증
        valid_files = []
        invalid_files = []

        for file in uploaded_files:
            is_valid, error_msg = validate_image_file(file)
            if is_valid:
                valid_files.append(file)
            else:
                invalid_files.append((file.name, error_msg))

        # 검증 결과 표시
        if invalid_files:
            st.error("⚠️ 일부 파일이 유효하지 않습니다:")
            for filename, error in invalid_files:
                st.warning(f"❌ {filename}: {error}")

        if valid_files:
            st.success(f"✅ {len(valid_files)}개의 이미지가 성공적으로 업로드되었습니다!")

            # 세션 상태에 저장
            st.session_state.uploaded_files = valid_files

            return valid_files

    # 업로더가 비어있고 세션 상태에 파일이 있으면 세션 상태 반환
    # (Remove 버튼으로 삭제 후 uploader가 리셋되었을 때)
    if st.session_state.get('uploaded_files'):
        return st.session_state.uploaded_files

    return None


def render_image_preview(uploaded_files: List, columns: int = 3):
    """
    이미지 미리보기 그리드 렌더링

    Args:
        uploaded_files: 업로드된 파일 리스트
        columns: 그리드의 열 개수
    """
    # 세션 상태에서 최신 파일 리스트 가져오기
    current_files = st.session_state.get('uploaded_files', uploaded_files)

    if not current_files:
        return

    st.markdown("---")
    st.markdown("### 🖼️ Image Preview")

    # 그리드 레이아웃 생성
    cols = st.columns(columns)

    for idx, file in enumerate(current_files):
        col_idx = idx % columns

        with cols[col_idx]:
            try:
                # 이미지 로드 및 표시
                img = Image.open(file)
                file.seek(0)

                # 이미지 표시
                st.image(img, width='stretch', caption=file.name)

                # 이미지 정보
                info = get_image_info(file)
                st.caption(f"📐 {info['width']}×{info['height']} | 💾 {info['size_mb']:.2f}MB")

                # 삭제 버튼
                if st.button(f"🗑️ Remove", key=f"remove_{file.name}_{idx}"):
                    # 세션 상태에서 해당 파일 제거
                    if 'uploaded_files' in st.session_state:
                        # 파일명으로 필터링하여 제거
                        st.session_state.uploaded_files = [
                            f for i, f in enumerate(st.session_state.uploaded_files)
                            if i != idx
                        ]
                        # 업로더 키 증가 (file_uploader 위젯 리셋)
                        st.session_state.uploader_key += 1
                        st.rerun()

            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")


def render_file_info_table(uploaded_files: List):
    """
    파일 정보 테이블 렌더링

    Args:
        uploaded_files: 업로드된 파일 리스트
    """
    # 세션 상태에서 최신 파일 리스트 가져오기
    current_files = st.session_state.get('uploaded_files', uploaded_files)

    if not current_files:
        return

    st.markdown("### 📋 File Information")

    # 데이터 준비
    data = []
    for idx, file in enumerate(current_files, 1):
        info = get_image_info(file)

        if 'error' not in info:
            data.append({
                '#': idx,
                'Filename': info['filename'],
                'Size': f"{info['size_mb']:.2f} MB",
                'Dimensions': f"{info['width']} × {info['height']}",
                'Format': info['format']
            })

    # 테이블 표시
    if data:
        st.dataframe(
            data,
            width='stretch',
            hide_index=True
        )


# ============================================================================
# 세션 상태 관리
# ============================================================================

def initialize_upload_state():
    """업로드 기능을 위한 세션 상태 초기화"""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    if 'upload_counter' not in st.session_state:
        st.session_state.upload_counter = 0

    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0


def clear_uploaded_files():
    """세션 상태에서 모든 업로드된 파일 제거"""
    if 'uploaded_files' in st.session_state:
        st.session_state.uploaded_files = []

    # 업로더 키 증가 (file_uploader 위젯 리셋)
    if 'uploader_key' in st.session_state:
        st.session_state.uploader_key += 1

    st.success("✅ 모든 파일이 제거되었습니다!")


def get_uploaded_files() -> List:
    """
    세션 상태에서 업로드된 파일 가져오기

    Returns:
        업로드된 파일 리스트
    """
    return st.session_state.get('uploaded_files', [])


# ============================================================================
# 완전한 업로드 위젯
# ============================================================================

def render_complete_uploader(preview_columns: int = 3, show_table: bool = True):
    """
    모든 기능을 포함한 완전한 업로드 위젯 렌더링

    Args:
        preview_columns: 미리보기 그리드의 열 개수
        show_table: 파일 정보 테이블 표시 여부
    """
    # 상태 초기화
    initialize_upload_state()

    # 업로더 렌더링
    uploaded_files = render_image_uploader()

    if uploaded_files:
        # 미리보기 그리드
        render_image_preview(uploaded_files, columns=preview_columns)

        st.markdown("---")

        # 파일 정보 테이블
        if show_table:
            render_file_info_table(uploaded_files)

        return uploaded_files
    else:
        st.info("👆 이미지를 업로드하세요")
        return None
