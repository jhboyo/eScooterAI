"""
웹캠 실시간 PPE 탐지 컴포넌트

streamlit-webrtc를 사용하여 실시간 비디오 스트리밍과 객체 탐지를 제공합니다.
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import cv2
import numpy as np
import threading
from collections import deque
import time
from gtts import gTTS
import pygame
import tempfile
import os


class VoiceAlertManager:
    """
    AI 음성 경고 시스템 매니저 (Streamlit 웹앱용)

    PPE 미착용 감지 시 한국어 음성 경고를 재생합니다.
    로컬 환경에서만 작동하며, Streamlit Cloud에서는 서버에서만 재생됩니다.
    """

    def __init__(self, cooldown_seconds: int = 10):
        """
        음성 경고 매니저 초기화

        Args:
            cooldown_seconds: 같은 경고의 재생 간격 (초, 기본값: 10초)
        """
        self.cooldown_seconds = cooldown_seconds  # 쿨다운 시간
        self.last_alert_time = {}  # 마지막 경고 시간 기록
        self.lock = threading.Lock()  # 스레드 안전성을 위한 락
        self.audio_cache = {}  # 생성된 음성 파일 캐시

        # pygame mixer 초기화 시도
        try:
            pygame.mixer.init()
            self.enabled = True
        except Exception as e:
            print(f"⚠️ 음성 경고 시스템 초기화 실패 (정상, Cloud 환경): {e}")
            self.enabled = False
    
    def _generate_audio(self, text: str, lang: str = 'ko') -> str:
        """
        텍스트를 음성 파일로 변환
        
        Args:
            text: 변환할 텍스트
            lang: 언어 코드 (기본값: 한국어)
            
        Returns:
            생성된 음성 파일 경로
        """
        # 캐시 확인
        cache_key = f"{text}_{lang}"
        if cache_key in self.audio_cache:
            return self.audio_cache[cache_key]
        
        try:
            # gTTS로 음성 생성
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # 임시 파일에 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_path = fp.name
                tts.save(temp_path)
            
            # 캐시에 저장
            self.audio_cache[cache_key] = temp_path
            return temp_path
            
        except Exception as e:
            print(f"⚠️ 음성 생성 실패: {e}")
            return None
    
    def play_alert(self, alert_type: str, force: bool = False):
        """
        음성 경고 재생 (로컬 환경에서만 작동)

        Args:
            alert_type: 경고 유형 ('helmet', 'vest', 'danger')
            force: True일 경우 쿨다운 무시하고 강제 재생
        """
        if not self.enabled:
            return

        # 쿨다운 체크
        with self.lock:
            current_time = time.time()
            last_time = self.last_alert_time.get(alert_type, 0)

            if not force and (current_time - last_time) < self.cooldown_seconds:
                return  # 쿨다운 중이므로 재생하지 않음

            self.last_alert_time[alert_type] = current_time

        # 경고 메시지 선택
        messages = {
            'helmet': '안전모를 착용하세요',
            'vest': '안전 조끼를 착용하세요',
            'danger': '위험! 안전 장비를 착용하세요'
        }

        message = messages.get(alert_type, '안전 수칙을 준수하세요')

        # 별도 스레드에서 재생 (메인 스레드 차단 방지)
        thread = threading.Thread(
            target=self._play_audio_thread,
            args=(message,),
            daemon=True
        )
        thread.start()

    def _play_audio_thread(self, text: str):
        """
        음성 재생 스레드 (내부 메서드)

        Args:
            text: 재생할 텍스트
        """
        try:
            audio_path = self._generate_audio(text)
            if audio_path and os.path.exists(audio_path):
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()

                # 재생이 끝날 때까지 대기
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

        except Exception as e:
            print(f"⚠️ 음성 재생 실패: {e}")
    
    def cleanup(self):
        """임시 파일 정리"""
        for path in self.audio_cache.values():
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass
        self.audio_cache.clear()


class PPEVideoProcessor(VideoProcessorBase):
    """
    실시간 비디오 프레임 처리 클래스
    
    웹캠에서 받은 각 프레임에 대해 YOLOv8 모델로 실시간 추론을 수행하고
    결과를 시각화하여 다시 브라우저로 전송합니다.
    """
    
    def __init__(self, model, conf_threshold: float = 0.55, iou_threshold: float = 0.45, 
                 enable_voice_alert: bool = True):
        """
        Args:
            model: YOLOv8 모델 객체
            conf_threshold: 신뢰도 임계값
            iou_threshold: IoU 임계값 (NMS)
            enable_voice_alert: 음성 경고 활성화 여부
        """
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 음성 경고 매니저 초기화
        self.voice_alert_enabled = enable_voice_alert
        if enable_voice_alert:
            self.voice_manager = VoiceAlertManager(cooldown_seconds=10)
        else:
            self.voice_manager = None
        
        # 클래스별 색상 정의 (BGR 형식 - OpenCV)
        self.class_colors = {
            0: (255, 0, 0),    # helmet - 파란색
            1: (0, 0, 255),    # head - 빨간색
            2: (0, 255, 255)   # vest - 노란색
        }
        
        self.class_names = {
            0: "Helmet",
            1: "Head",
            2: "Vest"
        }
        
        # 통계 정보 (스레드 안전)
        self.lock = threading.Lock()
        self.stats = {
            'helmet': 0,
            'head': 0,
            'vest': 0,
            'total_workers': 0,
            'helmet_rate': 0.0,
            'safety_level': 'Unknown',
            'fps': 0.0,
            'frame_count': 0
        }
        
        # FPS 계산을 위한 큐 (최근 30프레임의 처리 시간 저장)
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        프레임 수신 및 처리 (WebRTC 콜백)
        
        Args:
            frame: av.VideoFrame 객체 (브라우저에서 전송된 비디오 프레임)
            
        Returns:
            처리된 av.VideoFrame 객체 (바운딩 박스가 그려진 프레임)
        """
        # av.VideoFrame을 numpy 배열로 변환 (BGR 형식)
        img = frame.to_ndarray(format="bgr24")
        
        # FPS 계산
        current_time = time.time()
        fps = 1 / (current_time - self.last_time) if current_time > self.last_time else 0
        self.last_time = current_time
        self.fps_queue.append(fps)
        avg_fps = np.mean(self.fps_queue) if len(self.fps_queue) > 0 else 0
        
        # YOLOv8 추론 (verbose=False로 콘솔 출력 최소화)
        results = self.model(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        # 탐지 결과 파싱
        helmet_count = 0
        head_count = 0
        vest_count = 0
        
        # 바운딩 박스 그리기
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = results.boxes.conf.cpu().numpy()  # 신뢰도 점수
            classes = results.boxes.cls.cpu().numpy().astype(int)  # 클래스 ID
            
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                
                # 클래스별 카운팅
                if cls == 0:
                    helmet_count += 1
                elif cls == 1:
                    head_count += 1
                elif cls == 2:
                    vest_count += 1
                
                # 바운딩 박스 그리기
                color = self.class_colors.get(cls, (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                
                # 라벨 텍스트 생성
                label = f"{self.class_names[cls]}: {score:.2f}"
                
                # 라벨 배경 그리기
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    img,
                    (x1, y1 - text_height - 15),
                    (x1 + text_width + 10, y1),
                    color,
                    -1  # 채우기
                )
                
                # 라벨 텍스트 그리기
                cv2.putText(
                    img,
                    label,
                    (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # 흰색 텍스트
                    2
                )
        
        # 헬멧 착용률 계산
        total_workers = helmet_count + head_count
        helmet_rate = (helmet_count / total_workers * 100) if total_workers > 0 else 0
        
        # 안전 수준 평가 및 음성 경고 (서버에서 재생, 브라우저에는 들리지 않음)
        if total_workers > 0:
            if helmet_rate >= 90:
                safety_level = "Excellent"
                safety_color = (0, 255, 0)  # 녹색
            elif helmet_rate >= 70:
                safety_level = "Caution"
                safety_color = (0, 165, 255)  # 주황색
                # 음성 경고: 헬멧 미착용 감지
                if self.voice_manager and head_count > 0:
                    self.voice_manager.play_alert('helmet')
            else:
                safety_level = "Dangerous"
                safety_color = (0, 0, 255)  # 빨간색
                # 음성 경고: 위험 수준
                if self.voice_manager:
                    if head_count >= 2:
                        self.voice_manager.play_alert('danger')
                    elif head_count > 0:
                        self.voice_manager.play_alert('helmet')
        else:
            safety_level = "No Workers"
            safety_color = (128, 128, 128)  # 회색
        
        # 통계 업데이트 (스레드 안전)
        with self.lock:
            self.stats = {
                'helmet': helmet_count,
                'head': head_count,
                'vest': vest_count,
                'total_workers': total_workers,
                'helmet_rate': helmet_rate,
                'safety_level': safety_level,
                'fps': avg_fps,
                'frame_count': self.stats.get('frame_count', 0) + 1
            }
        
        # 화면에 통계 정보 오버레이
        overlay_y = 35
        overlay_x = 15
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        # 반투명 배경 (통계 영역)
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        # FPS 표시
        cv2.putText(
            img,
            f"FPS: {avg_fps:.1f}",
            (overlay_x, overlay_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
        overlay_y += 35
        
        # 탐지 수 표시
        cv2.putText(
            img,
            f"Helmet: {helmet_count} | Head: {head_count} | Vest: {vest_count}",
            (overlay_x, overlay_y),
            font,
            0.6,
            (255, 255, 255),
            font_thickness
        )
        overlay_y += 35
        
        # 헬멧 착용률 표시
        if total_workers > 0:
            cv2.putText(
                img,
                f"Workers: {total_workers} | Rate: {helmet_rate:.1f}%",
                (overlay_x, overlay_y),
                font,
                0.6,
                (255, 255, 255),
                font_thickness
            )
            overlay_y += 35
            
            # 안전 수준 표시
            cv2.putText(
                img,
                f"Safety: {safety_level}",
                (overlay_x, overlay_y),
                font,
                font_scale,
                safety_color,
                font_thickness + 1
            )
        
        # numpy 배열을 av.VideoFrame으로 변환하여 반환
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def get_stats(self):
        """현재 통계 정보 반환 (스레드 안전)"""
        with self.lock:
            return self.stats.copy()


def render_webcam_detector(model, conf_threshold: float, iou_threshold: float):
    """
    웹캠 실시간 탐지 UI 렌더링
    
    Args:
        model: YOLOv8 모델 객체
        conf_threshold: 신뢰도 임계값
        iou_threshold: IoU 임계값
    """
    st.header("📹 실시간 웹캠 모니터링")

    # 음성 경고 설정
    enable_voice = st.checkbox(
        "🔊 AI 음성 경고 활성화 (로컬 환경 전용)",
        value=True,
        help="로컬 환경에서 실행 시 헬멧 미착용 감지 시 음성 경고가 재생됩니다. Streamlit Cloud에서는 시각적 경고만 제공됩니다."
    )

    st.markdown("""
    노트북 카메라 또는 외부 웹캠을 사용하여 **실시간으로** PPE 탐지를 수행합니다.

    **✨ 특징:**
    - 🎥 **진짜 실시간 비디오 스트리밍** (25-30 FPS)
    - 🔍 **프레임 단위 객체 탐지** (Helmet, Head, Vest)
    - 📊 **실시간 통계 업데이트** (착용률, 안전 수준)
    - 🚨 **시각적 경고 시스템** (헬멧 미착용 시 화면 경고)
    - 🔊 **AI 음성 경고** (로컬 환경에서만 작동)
    - ⚡ **낮은 지연시간** (< 100ms)

    > 💡 **음성 경고 안내**:
    > - **로컬 환경** (localhost): 음성 경고가 스피커로 재생됩니다.
    > - **Streamlit Cloud**: 서버에 사운드 카드가 없어 음성 경고가 작동하지 않습니다. 시각적 경고만 제공됩니다.
    
    **🚀 사용 방법:**
    1. 아래 **"START"** 버튼을 클릭하세요
    2. 브라우저에서 **카메라 접근 권한**을 허용하세요
    3. 실시간 탐지 결과를 확인하세요
    4. **"STOP"** 버튼으로 중지할 수 있습니다
    """)
    
    # WebRTC 설정 (STUN 서버)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # VideoProcessor 클래스를 사용하여 각 프레임 처리
    class VideoProcessorFactory:
        def __init__(self):
            self.processor = None

        def __call__(self):
            self.processor = PPEVideoProcessor(
                model=model,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                enable_voice_alert=enable_voice  # 사용자가 선택한 음성 경고 설정
            )
            return self.processor
    
    factory = VideoProcessorFactory()
    
    # 웹캠 스트리머 시작
    ctx = webrtc_streamer(
        key="ppe-detection-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=factory,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30, "max": 30}
            },
            "audio": False
        },
        async_processing=True,
        sendback_audio=False,  # Disable audio to prevent RTX codec issues
    )
    
    # 실시간 통계 표시
    st.markdown("---")
    st.subheader("📊 실시간 통계")
    
    if ctx.state.playing:
        # 통계 표시 플레이스홀더
        stats_placeholder = st.empty()
        
        # 통계 업데이트 루프
        while ctx.state.playing:
            if factory.processor:
                stats = factory.processor.get_stats()
                
                with stats_placeholder.container():
                    # 메트릭 표시 (4열)
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🔵 Helmet", stats['helmet'])
                    
                    with col2:
                        st.metric("🔴 Head", stats['head'], 
                                delta="위험" if stats['head'] > 0 else None,
                                delta_color="inverse")
                    
                    with col3:
                        st.metric("🟡 Vest", stats['vest'])
                    
                    with col4:
                        st.metric("⚡ FPS", f"{stats['fps']:.1f}")
                    
                    # 안전 수준 표시
                    if stats['total_workers'] > 0:
                        st.markdown(f"### 👷 작업자: {stats['total_workers']}명")
                        st.markdown(f"### 📈 헬멧 착용률: {stats['helmet_rate']:.1f}%")

                        safety_level = stats['safety_level']
                        head_count = stats['head']

                        if safety_level == "Excellent":
                            st.success(f"✅ **안전 수준: {safety_level}**")

                        elif safety_level == "Caution":
                            # 주의 수준 - 강조된 경고
                            st.markdown("""
                                <div style="
                                    background-color: #FFA500;
                                    color: white;
                                    padding: 20px;
                                    border-radius: 10px;
                                    text-align: center;
                                    font-size: 24px;
                                    font-weight: bold;
                                    margin: 10px 0;
                                    border: 3px solid #FF8C00;
                                ">
                                    ⚠️ 주의: 헬멧 미착용자 감지됨
                                </div>
                            """, unsafe_allow_html=True)
                            st.warning(f"⚠️ **안전 수준: {safety_level}** - 헬멧 미착용: {head_count}명")

                        elif safety_level == "Dangerous":
                            # 위험 수준 - 깜빡이는 전체 화면 경고
                            st.markdown("""
                                <style>
                                @keyframes blink {
                                    0%, 50% { opacity: 1; }
                                    25%, 75% { opacity: 0.3; }
                                }
                                .danger-alert {
                                    animation: blink 1.5s infinite;
                                }
                                </style>
                                <div class="danger-alert" style="
                                    background: linear-gradient(135deg, #FF0000 0%, #CC0000 100%);
                                    color: white;
                                    padding: 30px;
                                    border-radius: 15px;
                                    text-align: center;
                                    font-size: 32px;
                                    font-weight: bold;
                                    margin: 10px 0;
                                    border: 5px solid #8B0000;
                                    box-shadow: 0 0 30px rgba(255,0,0,0.5);
                                ">
                                    🚨 위험! 즉시 안전 조치 필요 🚨
                                    <br>
                                    <span style="font-size: 24px;">헬멧 미착용자: {head_count}명</span>
                                </div>
                            """.format(head_count=head_count), unsafe_allow_html=True)
                            st.error(f"🚨 **안전 수준: {safety_level}** - 즉각적인 조치가 필요합니다!")

                            # 추가 경고 메시지
                            st.markdown("""
                                <div style="
                                    background-color: #FFEBEE;
                                    color: #C62828;
                                    padding: 15px;
                                    border-radius: 5px;
                                    border-left: 5px solid #C62828;
                                    margin: 10px 0;
                                ">
                                    <strong>⚠️ 안전 관리자에게 즉시 알림:</strong><br>
                                    • 작업 현장의 안전 수칙 위반 감지<br>
                                    • 헬멧 미착용자가 {head_count}명 확인됨<br>
                                    • 즉시 안전 장비 착용을 지시하세요
                                </div>
                            """.format(head_count=head_count), unsafe_allow_html=True)
                    else:
                        st.info("ℹ️ 작업자가 탐지되지 않았습니다")
                    
                    # 추가 정보
                    st.caption(f"처리된 프레임: {stats['frame_count']:,}개")
            
            # 0.5초마다 업데이트
            time.sleep(0.5)
    else:
        st.info("👆 위의 **START** 버튼을 클릭하여 실시간 모니터링을 시작하세요")
        
        # 도움말
        with st.expander("💡 문제 해결"):
            st.markdown("""
            **카메라가 작동하지 않는 경우:**
            - 브라우저에서 카메라 권한을 허용했는지 확인하세요
            - 다른 앱에서 카메라를 사용 중이지 않은지 확인하세요
            - HTTPS 연결인지 확인하세요 (로컬호스트는 HTTP 가능)
            
            **느린 프레임 레이트:**
            - 사이드바에서 신뢰도 임계값을 높여보세요 (0.6-0.7)
            - 브라우저를 재시작해보세요
            - GPU가 사용 가능한지 확인하세요
            
            **연결이 끊기는 경우:**
            - 네트워크 연결을 확인하세요
            - 방화벽 설정을 확인하세요
            """)
