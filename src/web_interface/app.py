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

# ============================================================================
# 환경 감지 및 프로젝트 루트 설정
# ============================================================================
# Hugging Face Spaces: app.py가 루트에 위치
# 로컬 개발: app.py가 src/web_interface/에 위치
current_file = Path(__file__).resolve()

# Hugging Face Spaces 환경 감지
# SPACE_ID 환경변수가 있거나 현재 디렉토리에 models 폴더가 있으면 배포 환경
if os.environ.get("SPACE_ID") or (current_file.parent / "models").exists():
    # Hugging Face Spaces 또는 배포 환경 (app.py가 프로젝트 루트에 위치)
    project_root = current_file.parent
else:
    # 로컬 개발 환경 (app.py가 src/web_interface/에 위치)
    # 상위 디렉토리 2단계 올라가서 프로젝트 루트 찾기
    project_root = current_file.parent.parent.parent

# Python 모듈 검색 경로에 프로젝트 루트 추가
sys.path.insert(0, str(project_root))

# .env 파일 로드 (중요: 다른 모듈 import 전에 먼저 로드해야 환경변수가 적용됨!)
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# ============================================================================
# 모듈 Import
# ============================================================================

# UI 컴포넌트
from components.uploader import render_complete_uploader  # 이미지 업로드 UI
from components.statistics import create_image_statistics_table  # 통계 테이블 생성

# 유틸리티 함수
from utils.inference import load_model, run_inference_batch, get_model_path, summarize_results  # 추론 관련
from utils.plotting import render_comparison_view  # 결과 시각화

# Telegram 알림 모듈
from src.alert.telegram_notifier import notifier  # Telegram Bot 알림 전송

# ============================================================================
# Page Configuration
# ============================================================================

# Streamlit 페이지 설정
st.set_page_config(
    page_title="Safety Vision AI - 딥러닝 기반 건설현장 안전 장비(PPE) 착용 모니터링 플랫폼",  # 브라우저 탭 제목
    page_icon="🏗️",  # 브라우저 탭 아이콘
    layout="wide",  # 넓은 레이아웃 사용
    initial_sidebar_state="expanded",  # 사이드바 기본 열림 상태
    menu_items={
        'Get Help': 'https://github.com/jhboyo/SafetyVisionAI',  # 도움말 링크
        'Report a bug': 'https://github.com/jhboyo/SafetyVisionAI/issues',  # 버그 제보 링크
        'About': """
        # Safety Vision AI

        **PPE Detection System** using YOLOv8

        Detects:
        - 🔵 Helmet (착용)
        - 🔴 Head (미착용)
        - 🟡 Vest (안전조끼)

        Version 0.0.1
        """  # About 정보
    }
)

# ============================================================================
# 커스텀 CSS 스타일 로드
# ============================================================================

def load_custom_css():
    """
    외부 CSS 파일 로드

    assets/styles.css 파일을 읽어서 Streamlit 앱에 적용
    """
    # CSS 파일 경로 설정 (현재 파일 기준 assets/styles.css)
    css_file = Path(__file__).parent / "assets" / "styles.css"

    # CSS 파일 존재 여부 확인 및 적용
    if css_file.exists():
        # UTF-8 인코딩으로 CSS 파일 읽기
        with open(css_file, 'r', encoding='utf-8') as f:
            css_content = f.read()
            # HTML 스타일 태그로 감싸서 Streamlit에 적용
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        # CSS 파일이 없을 경우 경고 메시지 표시
        st.warning("⚠️ CSS 파일을 찾을 수 없습니다.")

# ============================================================================
# 사이드바 설정
# ============================================================================

def sidebar_config():
    """
    사이드바 설정 및 정보 표시

    Returns:
        dict: 모델 경로, 추론 파라미터 등 설정값 딕셔너리
    """
    with st.sidebar:
        st.header("⚙️ 설정")

        # 모델 선택 (best.pt: 최고 성능 모델, last.pt: 마지막 체크포인트)
        model_option = st.selectbox(
            "모델 선택",
            ["Best Model (best.pt)", "Last Checkpoint (last.pt)"],
            index=0  # 기본값: Best Model
        )

        # 신뢰도 임계값 설정 (0~1 사이의 값, 높을수록 엄격한 탐지)
        conf_threshold = st.slider(
            "신뢰도 임계값",
            min_value=0.1,
            max_value=1.0,
            value=0.55,  # 기본값: 0.55
            step=0.05,
            help="탐지를 위한 최소 신뢰도 점수 (높을수록 정확도 높지만 탐지 수 감소)"
        )

        # 고급 옵션 (접을 수 있는 섹션)
        with st.expander("🔧 고급 옵션"):
            # IoU (Intersection over Union) 임계값 설정
            # 중복 탐지 제거를 위한 NMS(Non-Maximum Suppression) 알고리즘에 사용
            iou_threshold = st.slider(
                "IoU 임계값",
                min_value=0.1,
                max_value=1.0,
                value=0.45,  # 기본값: 0.45
                step=0.05,
                help="NMS(Non-Maximum Suppression)를 위한 IoU 임계값 (중복 탐지 제거 기준)"
            )

            # 이미지당 최대 탐지 개수 제한
            max_det = st.number_input(
                "최대 탐지 개수",
                min_value=1,
                max_value=1000,
                value=300,  # 기본값: 300개
                step=10,
                help="이미지당 최대 탐지 객체 수 (너무 많은 탐지 방지)"
            )

            # 디버그 모드 활성화 여부
            debug_mode = st.checkbox(
                "디버그 모드",
                value=False,  # 기본값: 비활성화
                help="클래스별 탐지 정보 및 신뢰도 점수 상세 표시"
            )

        st.markdown("---")

        # Telegram 알림 설정 섹션
        st.header("📱 알림 설정")

        # Telegram 알림 활성화 여부에 따라 다른 UI 표시
        if notifier.enabled:
            # 알림 활성화 상태
            st.success("✅ Telegram 알림 활성화")
            st.caption(f"🤖 Bot: @SafetyVisionAI_Bot")
            st.caption(f"💬 Chat ID: {notifier.chat_id}")

            # 알림 발송 조건 안내
            with st.expander("⚙️ 알림 발송 조건"):
                st.markdown("""
                **알림이 전송되는 경우:**
                - 🔴 헬멧 미착용자 **2명 이상** 감지
                - 📊 헬멧 착용률 **80% 미만**

                (둘 중 하나만 만족해도 알림 전송)
                """)

            # Telegram Bot 연결 테스트 버튼
            if st.button("🔔 연결 테스트", help="Telegram Bot 연결 상태를 확인합니다"):
                with st.spinner("테스트 중..."):
                    # 실제 Telegram API로 테스트 메시지 전송
                    if notifier.test_connection():
                        st.success("✅ Telegram 연결 성공!")
                    else:
                        st.error("❌ Telegram 연결 실패. Bot Token과 Chat ID를 확인하세요.")
        else:
            # 알림 비활성화 상태
            st.info("ℹ️ Telegram 알림이 비활성화되어 있습니다")
            st.caption("`.env` 파일에서 `TELEGRAM_ALERTS_ENABLED=true` 설정 필요")

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

    # 선택한 모델 옵션에 따라 모델 파일명 결정
    model_filename = 'best.pt' if 'Best' in model_option else 'last.pt'
    # 모델 파일의 전체 경로 생성 (프로젝트루트/models/ppe_detection/weights/모델파일)
    model_path = project_root / 'models' / 'ppe_detection' / 'weights' / model_filename

    # 사용자가 설정한 모든 값을 딕셔너리로 반환 (메인 함수에서 사용)
    return {
        'model': model_filename,  # 모델 파일명 (best.pt 또는 last.pt)
        'model_path': str(model_path),  # 모델 전체 경로 (문자열)
        'conf': conf_threshold,  # 신뢰도 임계값
        'iou': iou_threshold,  # IoU 임계값 (NMS용)
        'max_det': max_det,  # 최대 탐지 개수
        'debug': debug_mode  # 디버그 모드 활성화 여부
    }

# ============================================================================
# 세션 상태 초기화
# ============================================================================

def init_session_state():
    """
    Streamlit 세션 상태 변수 초기화

    세션 상태는 페이지 새로고침 없이 데이터를 유지하기 위해 사용
    """
    # 업로드된 파일 리스트 (파일 업로더에서 관리)
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    # 추론 결과 저장 (탐지된 객체 정보, 이미지 등)
    if 'inference_results' not in st.session_state:
        st.session_state.inference_results = None

    # 추론에 소요된 시간 (초 단위)
    if 'inference_time' not in st.session_state:
        st.session_state.inference_time = 0

    # 추론 속도 (FPS, Frames Per Second)
    if 'inference_fps' not in st.session_state:
        st.session_state.inference_fps = 0

    # 모델 로드 완료 여부 플래그
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
    """
    메인 애플리케이션 로직

    Streamlit 앱의 전체 흐름을 제어하는 메인 함수
    """
    # 세션 상태 초기화 (앱 실행 시 최초 1회)
    init_session_state()

    # 커스텀 CSS 스타일 로드 (UI 디자인 적용)
    load_custom_css()

    # 페이지 상단 헤더 렌더링
    render_header()

    # 사이드바에서 사용자 설정값 가져오기 (모델, 임계값 등)
    settings = sidebar_config()

    # 메인 컨텐츠 영역 - 이미지 업로드 섹션
    # 3열로 미리보기, 업로드 테이블 표시
    uploaded_files = render_complete_uploader(preview_columns=3, show_table=True)

    # 업로드된 이미지가 있는 경우
    if uploaded_files:
        st.markdown("---")

        # 탐지 시작 버튼 (화면 중앙 배치)
        col1, col2, col3 = st.columns([1, 2, 1])  # 1:2:1 비율로 3열 생성
        with col2:  # 중앙 열에 버튼 배치
            if st.button("🚀 탐지 시작", width='stretch', type="primary"):
                # YOLOv8 모델 로드 (버튼 클릭 시점에 로드하여 메모리 효율성 증가)
                model_path = get_model_path(settings['model'])
                model = load_model(str(model_path))

                # 모델 로드 실패 시 에러 메시지 표시 후 중단
                if model is None:
                    st.error("❌ 모델 로드에 실패했습니다. 페이지를 새로고침하거나 관리자에게 문의하세요.")
                    st.stop()  # 더 이상 진행하지 않음

                # 세션 상태에서 업로드된 이미지 가져오기
                if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
                    uploaded_files = st.session_state.uploaded_files

                    # Streamlit UploadedFile 객체를 PIL Image 객체로 변환
                    from PIL import Image
                    import time
                    images = []  # PIL Image 리스트
                    filenames = []  # 파일명 리스트
                    for file in uploaded_files:
                        file.seek(0)  # 파일 포인터를 처음으로 이동 (중요!)
                        img = Image.open(file)  # PIL Image로 변환
                        images.append(img)
                        filenames.append(file.name)

                    # 추론 시간 측정 시작 (성능 분석용)
                    start_time = time.time()

                    # 배치 추론 실행 (여러 이미지를 한 번에 처리)
                    # show_progress=True로 진행바 표시
                    results = run_inference_batch(
                        model=model,  # YOLOv8 모델
                        images=images,  # PIL Image 리스트
                        conf=settings['conf'],  # 신뢰도 임계값
                        iou=settings['iou'],  # IoU 임계값 (NMS용)
                        max_det=settings['max_det'],  # 최대 탐지 개수
                        show_progress=True,  # Streamlit 진행바 표시
                        debug=settings['debug']  # 디버그 정보 포함 여부
                    )

                    # 추론 시간 측정 종료
                    total_time = time.time() - start_time
                    # 평균 FPS 계산 (초당 처리 이미지 수)
                    avg_fps = len(images) / total_time if total_time > 0 else 0

                    # 각 결과에 파일명과 원본 이미지 추가 (시각화용)
                    for i, result in enumerate(results):
                        result['filename'] = filenames[i]  # 파일명 추가
                        result['original_image'] = images[i]  # 원본 이미지 추가

                    # 추론 결과를 세션 상태에 저장 (페이지 새로고침 없이 유지)
                    st.session_state.inference_results = results
                    st.session_state.inference_time = total_time
                    st.session_state.inference_fps = avg_fps

                    # 추론 완료 메시지 표시
                    st.success("✅ 추론 완료!")

                    # 사용자가 설정한 추론 파라미터 표시
                    st.caption(f"**추론 설정**: 신뢰도 임계값 {settings['conf']:.2f}, IoU 임계값 {settings['iou']:.2f}")

                    # 전체 이미지에서 탐지된 객체 수 계산
                    total_detections = sum(len(r['detections']) for r in results)
                    st.info(f"📊 **전체 {len(results)}개 이미지에서 총 {total_detections}개 객체 탐지됨**")

                    # 클래스별 탐지 수 집계 (helmet, head, vest)
                    all_class_count = {}
                    for result in results:
                        for det in result['detections']:
                            cls = det['class_name']  # 클래스명 (helmet, head, vest)
                            all_class_count[cls] = all_class_count.get(cls, 0) + 1

                    # 각 클래스별 탐지 수 추출
                    helmet_total = all_class_count.get('helmet', 0)  # 헬멧 착용
                    head_total = all_class_count.get('head', 0)  # 헬멧 미착용
                    vest_total = all_class_count.get('vest', 0)  # 안전조끼
                    person_total = helmet_total + head_total  # 전체 작업자 수 (Person = Helmet + Head)

                    # 4열 레이아웃으로 주요 통계 표시
                    total_cols = st.columns(4)
                    with total_cols[0]:
                        # 전체 작업자 수 (헬멧 착용 + 미착용)
                        st.metric("👷 전체 Person", f"{person_total}명")
                    with total_cols[1]:
                        # 헬멧 착용자 수
                        st.metric("🔵 전체 Helmet", helmet_total)
                    with total_cols[2]:
                        # 헬멧 미착용자 수 (위험 요소)
                        if head_total > 0:
                            # 미착용자가 있으면 빨간색으로 경고 표시
                            st.metric("🔴 전체 Head (미착용)", head_total, delta="⚠️ 위험", delta_color="inverse")
                        else:
                            # 미착용자가 없으면 안전 표시
                            st.metric("🔴 전체 Head (미착용)", 0, delta="✅ 안전", delta_color="normal")
                    with total_cols[3]:
                        # 안전조끼 착용 수
                        st.metric("🟡 전체 Vest", vest_total)

                    # Telegram 알림 전송 로직
                    # 조건: 헬멧 미착용자 2명 이상 OR 착용률 80% 미만
                    if notifier.enabled and person_total > 0:
                        # 헬멧 착용률 계산 (퍼센트)
                        helmet_rate = (helmet_total / person_total * 100) if person_total > 0 else 0

                        # 위험 상황 감지 시 알림 전송
                        # 조건 1: 헬멧 미착용자가 2명 이상
                        # 조건 2: 헬멧 착용률이 80% 미만
                        if head_total >= 2 or helmet_rate < 80:
                            # 첫 번째 이미지의 탐지 결과를 알림 이미지로 사용
                            alert_image = results[0].get('annotated_image') if results else None

                            # Telegram 알림 전송 (스피너로 로딩 표시)
                            with st.spinner("📱 Telegram 알림 전송 중..."):
                                success = notifier.send_safety_alert(
                                    head_count=head_total,  # 미착용자 수
                                    total_workers=person_total,  # 전체 작업자 수
                                    helmet_rate=helmet_rate,  # 착용률 (%)
                                    image=alert_image,  # 탐지 결과 이미지
                                    location="건설 현장"  # 현장 위치
                                )

                                # 전송 결과 표시
                                if success:
                                    st.success("📱 Telegram 알림이 전송되었습니다!")
                                else:
                                    st.warning("⚠️ Telegram 알림 전송 실패")

                    st.info("💡 아래로 스크롤하여 각 이미지별 상세 탐지 결과를 확인하세요.")

                else:
                    st.error("❌ 업로드된 이미지를 찾을 수 없습니다.")

    # ============================================================================
    # 추론 결과 표시 섹션
    # ============================================================================
    # 세션 상태에 저장된 추론 결과가 있으면 표시
    if st.session_state.inference_results:
        st.markdown("---")

        # 전체 결과 요약 통계 계산 (유틸리티 함수 사용)
        summary = summarize_results(st.session_state.inference_results)

        # 클래스별 탐지 수 추출
        helmet_count = summary['class_counts'].get('helmet', 0)  # 헬멧 착용
        head_count = summary['class_counts'].get('head', 0)  # 헬멧 미착용
        vest_count = summary['class_counts'].get('vest', 0)  # 안전조끼
        total_workers = helmet_count + head_count  # 전체 작업자
        # 헬멧 착용률 계산 (백분율)
        helmet_rate = (helmet_count / total_workers * 100) if total_workers > 0 else 0

        # 안전 수준 평가 (착용률 기준)
        # 가장 중요한 정보를 최상단에 크게 표시
        st.markdown("### ✅ 탐지 완료")

        if total_workers > 0:
            # 착용률 90% 이상: 우수 (Excellent)
            if helmet_rate >= 90:
                st.success(f"""
                ### 🛡️ 안전 수준: **Excellent** ✅
                **헬멧 착용률: {helmet_rate:.1f}%** (매우 안전합니다)
                """)
            # 착용률 70~90%: 주의 (Caution)
            elif helmet_rate >= 70:
                st.warning(f"""
                ### 🛡️ 안전 수준: **Caution** ⚠️
                **헬멧 착용률: {helmet_rate:.1f}%** (주의가 필요합니다)
                """)
            # 착용률 70% 미만: 위험 (Dangerous)
            else:
                st.error(f"""
                ### 🛡️ 안전 수준: **Dangerous** 🚨
                **헬멧 착용률: {helmet_rate:.1f}%** (위험 상태입니다!)
                """)
        else:
            # 작업자가 탐지되지 않은 경우
            st.info("### ℹ️ 작업자가 탐지되지 않았습니다")

        # 주요 통계 메트릭 (3열 레이아웃으로 간결하게 표시)
        col1, col2, col3 = st.columns(3)

        with col1:
            # 헬멧 착용자 수 (안전 표시)
            st.metric("🔵 헬멧 착용", f"{helmet_count}명",
                     delta="안전" if helmet_count > 0 else None,
                     delta_color="normal")

        with col2:
            # 헬멧 미착용자 수 (위험 표시)
            st.metric("🔴 헬멧 미착용", f"{head_count}명",
                     delta="위험" if head_count > 0 else None,
                     delta_color="inverse")

        with col3:
            # 안전조끼 착용 수
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
                        width='stretch',
                        hide_index=True
                    )

                    # Head 클래스 필터링
                    head_detections = [d for d in all_detections_detail if d['class'] == 'head']
                    if head_detections:
                        st.markdown("#### 🔴 헬멧 미착용 (Head) 탐지 상세")
                        st.dataframe(
                            head_detections,
                            width='stretch',
                            hide_index=True
                        )
                    else:
                        st.info("✅ Head 클래스 탐지 없음 (모두 헬멧을 착용하고 있습니다)")

        # 이미지별 탐지 결과 시각화 (원본 이미지 vs 탐지 결과 비교)
        if st.session_state.get('uploaded_files') and st.session_state.get('inference_results'):
            render_comparison_view(
                # 모든 결과에서 원본 이미지 추출
                original_images=[result['original_image'] for result in st.session_state.inference_results],
                # 추론 결과 (탐지된 객체 정보 포함)
                results=st.session_state.inference_results,
                # 업로드된 파일 정보
                uploaded_files=st.session_state.get('uploaded_files_info', st.session_state.uploaded_files)
            )

        # ============================================================================
        # 이미지별 통계 테이블 (화면 제일 하단)
        # ============================================================================

        st.markdown("---")
        st.markdown("## 📋 이미지별 상세 통계")
        st.caption("각 이미지의 탐지 결과를 표로 확인합니다")

        # 추론 결과를 표 형식으로 변환 (유틸리티 함수 사용)
        stats_table = create_image_statistics_table(st.session_state.inference_results)

        # Streamlit 데이터프레임으로 표 렌더링
        st.dataframe(
            stats_table,  # 통계 테이블 데이터
            width='stretch',  # 화면 전체 너비 사용
            hide_index=True,  # 인덱스 열 숨기기
            # 각 열의 너비 및 타입 설정
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

    # ============================================================================
    # 페이지 하단 Footer
    # ============================================================================
    st.markdown("""
        <div class="footer">
            <p>Safety Vision AI v1.0.0 | Built with Streamlit & YOLOv8</p>
            <p>© 2025 Safety Vision AI Team</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# Application Entry Point (프로그램 시작점)
# ============================================================================

if __name__ == "__main__":
    # Python 스크립트로 직접 실행될 때만 main() 함수 호출
    # 모듈로 import될 때는 실행되지 않음
    main()
