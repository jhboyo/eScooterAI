"""
Safety Vision AI - PPE Detection Web Dashboard

YOLOv8 모델을 사용한 개인보호구(PPE) 탐지 웹 인터페이스
건설현장에서 헬멧, 헬멧 미착용, 안전조끼를 탐지합니다.

Author: Safety Vision AI Team
Date: 2025-11-22
"""

import streamlit as st
from pathlib import Path
import sys
import os

# 환경 감지 및 프로젝트 루트 설정
# Hugging Face Spaces: app.py가 루트에 위치
# 로컬 개발: app.py가 src/5_web_interface/에 위치
current_file = Path(__file__).resolve()

# Hugging Face Spaces 환경 감지
if os.environ.get("SPACE_ID") or (current_file.parent / "models").exists():
    # Hugging Face Spaces 또는 배포 환경
    project_root = current_file.parent
else:
    # 로컬 개발 환경
    project_root = current_file.parent.parent.parent

sys.path.insert(0, str(project_root))

# Import components
from components.uploader import render_complete_uploader
from components.statistics import create_image_statistics_table

# Import utilities
from utils.inference import load_model, run_inference_batch, get_model_path, summarize_results
from utils.plotting import render_comparison_view

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Safety Vision AI - 딥러닝 기반 건설현장 안전 장비(PPE) 착용 모니터링 플랫폼",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/jhboyo/SafetyVisionAI',
        'Report a bug': 'https://github.com/jhboyo/SafetyVisionAI/issues',
        'About': """
        # Safety Vision AI

        **PPE Detection System** using YOLOv8

        Detects:
        - 🔵 Helmet (착용)
        - 🔴 Head (미착용)
        - 🟡 Vest (안전조끼)

        Version 0.0.1
        """
    }
)

# ============================================================================
# 커스텀 CSS 스타일 로드
# ============================================================================

def load_custom_css():
    """외부 CSS 파일 로드"""
    # CSS 파일 경로
    css_file = Path(__file__).parent / "assets" / "styles.css"

    # CSS 파일 읽기 및 적용
    if css_file.exists():
        with open(css_file, 'r', encoding='utf-8') as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        # CSS 파일이 없을 경우 경고 메시지
        st.warning("⚠️ CSS 파일을 찾을 수 없습니다.")

# ============================================================================
# 사이드바 설정
# ============================================================================

def sidebar_config():
    """사이드바 설정 및 정보 표시"""
    with st.sidebar:
        st.header("⚙️ 설정")

        # 모델 선택
        model_option = st.selectbox(
            "모델 선택",
            ["Best Model (best.pt)", "Last Checkpoint (last.pt)"],
            index=0
        )

        # 신뢰도 임계값 설정
        conf_threshold = st.slider(
            "신뢰도 임계값",
            min_value=0.1,
            max_value=1.0,
            value=0.55,
            step=0.05,
            help="탐지를 위한 최소 신뢰도 점수"
        )

        # 고급 옵션
        with st.expander("🔧 고급 옵션"):
            # IoU 임계값 설정
            iou_threshold = st.slider(
                "IoU 임계값",
                min_value=0.1,
                max_value=1.0,
                value=0.45,
                step=0.05,
                help="NMS(Non-Maximum Suppression)를 위한 IoU 임계값"
            )

            # 최대 탐지 개수 설정
            max_det = st.number_input(
                "최대 탐지 개수",
                min_value=1,
                max_value=1000,
                value=300,
                step=10,
                help="이미지당 최대 탐지 객체 수"
            )

            # 디버그 모드 설정
            debug_mode = st.checkbox(
                "디버그 모드",
                value=False,
                help="클래스별 탐지 정보 및 신뢰도 점수 표시"
            )

        st.markdown("---")

        # 정보 섹션
        st.header("ℹ️ 정보")

        st.markdown("""
        ### 객체 탐지 클래스
        - 👷 **Person**: 전체 작업자 (Helmet + Head)
        - 🔵 **Helmet**: 헬멧 착용
        - 🔴 **Head**: 헬멧 미착용 (위험!)
        - 🟡 **Vest**: 안전조끼 착용

        ### 안전 수준
        - ✅ **Excellent**: ≥ 90% 헬멧 착용률
        - ⚠️ **Caution**: 70-89% 헬멧 착용률
        - 🚨 **Dangerous**: < 70% 헬멧 착용률
        """)

        st.markdown("---")

        # 팀 정보
        st.markdown("""
        ### 👥 팀
        Safety Vision AI
        김상진, 김준호, 김한솔, 유승근, 홍준재
        """)

    # 모델 경로 설정
    model_filename = 'best.pt' if 'Best' in model_option else 'last.pt'
    model_path = project_root / 'models' / 'ppe_detection' / 'weights' / model_filename

    # 설정값 반환
    return {
        'model': model_filename,
        'model_path': str(model_path),
        'conf': conf_threshold,
        'iou': iou_threshold,
        'max_det': max_det,
        'debug': debug_mode
    }

# ============================================================================
# 세션 상태 초기화
# ============================================================================

def init_session_state():
    """세션 상태 변수 초기화"""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'inference_results' not in st.session_state:
        st.session_state.inference_results = None
    if 'inference_time' not in st.session_state:
        st.session_state.inference_time = 0
    if 'inference_fps' not in st.session_state:
        st.session_state.inference_fps = 0
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

# ============================================================================
# 메인 헤더
# ============================================================================

def render_header():
    """메인 헤더 렌더링"""
    st.markdown("""
        <div class="main-header">
            <h1>🏗️ Safety Vision AI</h1>
            <p>딥러닝 기반 건설현장 안전 장비(PPE) 착용 모니터링 플랫폼</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# 메인 컨텐츠
# ============================================================================

def main():
    """메인 애플리케이션 로직"""
    # 세션 상태 초기화
    init_session_state()

    # 커스텀 CSS 로드
    load_custom_css()

    # 헤더 렌더링
    render_header()

    # 사이드바 설정 가져오기
    settings = sidebar_config()

    # 모델 로드 (캐싱됨)
    model_path = get_model_path(settings['model'])
    model = load_model(str(model_path))

    # 메인 컨텐츠 영역 - 이미지 업로드 섹션
    uploaded_files = render_complete_uploader(preview_columns=3, show_table=True)

    # 업로드된 이미지가 있고 모델이 로드된 경우
    if uploaded_files and model is not None:
        st.markdown("---")

        # 탐지 시작 버튼
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 탐지 시작", use_container_width=True, type="primary"):
                # 세션에서 업로드된 이미지 가져오기
                if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
                    uploaded_files = st.session_state.uploaded_files

                    # UploadedFile을 PIL Image로 변환
                    from PIL import Image
                    import time
                    images = []
                    filenames = []
                    for file in uploaded_files:
                        file.seek(0)  # 파일 포인터를 처음으로 이동
                        img = Image.open(file)
                        images.append(img)
                        filenames.append(file.name)

                    # 추론 시간 측정 시작
                    start_time = time.time()

                    # 배치 추론 실행 (진행바는 show_progress로 표시됨)
                    results = run_inference_batch(
                        model=model,
                        images=images,
                        conf=settings['conf'],
                        iou=settings['iou'],
                        max_det=settings['max_det'],
                        show_progress=True,
                        debug=settings['debug']
                    )

                    # 추론 시간 측정 종료
                    total_time = time.time() - start_time
                    avg_fps = len(images) / total_time if total_time > 0 else 0

                    # 결과에 파일명 추가
                    for i, result in enumerate(results):
                        result['filename'] = filenames[i]
                        result['original_image'] = images[i]

                    # 결과를 세션 상태에 저장
                    st.session_state.inference_results = results
                    st.session_state.inference_time = total_time
                    st.session_state.inference_fps = avg_fps

                    # 추론 완료 후 간단한 요약만 표시
                    st.success("✅ 추론 완료!")

                    # 추론 설정 정보
                    st.caption(f"**추론 설정**: 신뢰도 임계값 {settings['conf']:.2f}, IoU 임계값 {settings['iou']:.2f}")

                    # 전체 요약
                    total_detections = sum(len(r['detections']) for r in results)
                    st.info(f"📊 **전체 {len(results)}개 이미지에서 총 {total_detections}개 객체 탐지됨**")

                    # 전체 클래스별 통계만 간단히 표시
                    all_class_count = {}
                    for result in results:
                        for det in result['detections']:
                            cls = det['class_name']
                            all_class_count[cls] = all_class_count.get(cls, 0) + 1

                    helmet_total = all_class_count.get('helmet', 0)
                    head_total = all_class_count.get('head', 0)
                    vest_total = all_class_count.get('vest', 0)
                    person_total = helmet_total + head_total  # Person = Helmet + Head

                    total_cols = st.columns(4)
                    with total_cols[0]:
                        st.metric("👷 전체 Person", f"{person_total}명")
                    with total_cols[1]:
                        st.metric("🔵 전체 Helmet", helmet_total)
                    with total_cols[2]:
                        if head_total > 0:
                            st.metric("🔴 전체 Head (미착용)", head_total, delta="⚠️ 위험", delta_color="inverse")
                        else:
                            st.metric("🔴 전체 Head (미착용)", 0, delta="✅ 안전", delta_color="normal")
                    with total_cols[3]:
                        st.metric("🟡 전체 Vest", vest_total)

                    st.info("💡 아래로 스크롤하여 각 이미지별 상세 탐지 결과를 확인하세요.")

                else:
                    st.error("❌ 업로드된 이미지를 찾을 수 없습니다.")

    # 추론 결과 표시
    if st.session_state.inference_results:
        st.markdown("---")

        # 요약 통계 계산
        summary = summarize_results(st.session_state.inference_results)

        helmet_count = summary['class_counts'].get('helmet', 0)
        head_count = summary['class_counts'].get('head', 0)
        vest_count = summary['class_counts'].get('vest', 0)
        total_workers = helmet_count + head_count
        helmet_rate = (helmet_count / total_workers * 100) if total_workers > 0 else 0

        # 안전 수준 평가 (가장 중요한 정보를 최상단에 크게 표시)
        st.markdown("### ✅ 탐지 완료")

        if total_workers > 0:
            if helmet_rate >= 90:
                st.success(f"""
                ### 🛡️ 안전 수준: **Excellent** ✅
                **헬멧 착용률: {helmet_rate:.1f}%** (매우 안전합니다)
                """)
            elif helmet_rate >= 70:
                st.warning(f"""
                ### 🛡️ 안전 수준: **Caution** ⚠️
                **헬멧 착용률: {helmet_rate:.1f}%** (주의가 필요합니다)
                """)
            else:
                st.error(f"""
                ### 🛡️ 안전 수준: **Dangerous** 🚨
                **헬멧 착용률: {helmet_rate:.1f}%** (위험 상태입니다!)
                """)
        else:
            st.info("### ℹ️ 작업자가 탐지되지 않았습니다")

        # 주요 통계 (간결하게 3개만)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🔵 헬멧 착용", f"{helmet_count}명",
                     delta="안전" if helmet_count > 0 else None,
                     delta_color="normal")

        with col2:
            st.metric("🔴 헬멧 미착용", f"{head_count}명",
                     delta="위험" if head_count > 0 else None,
                     delta_color="inverse")

        with col3:
            st.metric("🟡 안전조끼", f"{vest_count}개",
                     delta=None)

        # 상세 정보는 접을 수 있게
        with st.expander("📊 상세 통계 보기"):
            detail_col1, detail_col2, detail_col3, detail_col4, detail_col5 = st.columns(5)

            with detail_col1:
                st.metric("📸 처리 이미지", f"{summary['total_images']}개")

            with detail_col2:
                st.metric("🎯 총 탐지", f"{summary['total_detections']}개")

            with detail_col3:
                st.metric("📊 평균 탐지", f"{summary['avg_detections_per_image']:.1f}개/이미지")

            with detail_col4:
                total_time = st.session_state.get('inference_time', 0)
                st.metric("⏱️ 총 소요 시간", f"{total_time:.2f}초")

            with detail_col5:
                fps = st.session_state.get('inference_fps', 0)
                st.metric("⚡ FPS", f"{fps:.1f}")

        # 디버그 정보 표시
        if settings.get('debug') and st.session_state.inference_results:
            with st.expander("🔍 디버그 정보 (클래스별 탐지 상세)"):
                # 전체 클래스 분포 집계
                total_class_dist = {'helmet': 0, 'head': 0, 'vest': 0}
                all_detections_detail = []

                for result in st.session_state.inference_results:
                    if 'debug_info' in result:
                        debug_info = result['debug_info']
                        # 클래스 분포 누적
                        for cls_name, count in debug_info['class_distribution'].items():
                            if cls_name in total_class_dist:
                                total_class_dist[cls_name] += count

                        # 모든 탐지 정보 수집
                        for det in debug_info['all_detections']:
                            all_detections_detail.append({
                                'filename': result.get('filename', 'Unknown'),
                                'class': det['class_name'],
                                'confidence': f"{det['confidence']:.3f}",
                                'bbox': f"({det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f})"
                            })

                # 클래스별 탐지 수 표시
                st.markdown("#### 📊 클래스별 탐지 수")
                debug_col1, debug_col2, debug_col3 = st.columns(3)

                with debug_col1:
                    st.metric("🔵 Helmet", total_class_dist['helmet'])
                with debug_col2:
                    st.metric("🔴 Head", total_class_dist['head'],
                             delta="⚠️ 미착용" if total_class_dist['head'] > 0 else None)
                with debug_col3:
                    st.metric("🟡 Vest", total_class_dist['vest'])

                # 모든 탐지 상세 정보 테이블
                if all_detections_detail:
                    st.markdown("#### 📋 모든 탐지 상세 정보")
                    st.dataframe(
                        all_detections_detail,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Head 클래스 필터링
                    head_detections = [d for d in all_detections_detail if d['class'] == 'head']
                    if head_detections:
                        st.markdown("#### 🔴 헬멧 미착용 (Head) 탐지 상세")
                        st.dataframe(
                            head_detections,
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("✅ Head 클래스 탐지 없음 (모두 헬멧을 착용하고 있습니다)")

        # 이미지별 탐지 결과 시각화
        if st.session_state.get('uploaded_files') and st.session_state.get('inference_results'):
            render_comparison_view(
                original_images=[result['original_image'] for result in st.session_state.inference_results],
                results=st.session_state.inference_results,
                uploaded_files=st.session_state.get('uploaded_files_info', st.session_state.uploaded_files)
            )

        # ============================================================================
        # 이미지별 통계 테이블 (화면 제일 하단)
        # ============================================================================

        st.markdown("---")
        st.markdown("## 📋 이미지별 상세 통계")
        st.caption("각 이미지의 탐지 결과를 표로 확인합니다")

        stats_table = create_image_statistics_table(st.session_state.inference_results)
        st.dataframe(
            stats_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                '번호': st.column_config.NumberColumn('번호', width='small'),
                '이미지 파일': st.column_config.TextColumn('이미지 파일', width='large'),
                '🔵 Helmet': st.column_config.NumberColumn('🔵 Helmet', width='small'),
                '🔴 Head': st.column_config.NumberColumn('🔴 Head', width='small'),
                '🟡 Vest': st.column_config.NumberColumn('🟡 Vest', width='small'),
                '👷 Person': st.column_config.NumberColumn('👷 Person', width='small'),
                '착용률 (%)': st.column_config.TextColumn('착용률 (%)', width='small'),
                '안전 수준': st.column_config.TextColumn('안전 수준', width='medium')
            }
        )

    # Footer
    st.markdown("""
        <div class="footer">
            <p>Safety Vision AI v1.0.0 | Built with Streamlit & YOLOv8</p>
            <p>© 2025 Safety Vision AI Team</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
