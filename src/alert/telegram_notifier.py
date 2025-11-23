"""
Telegram Bot 알림 모듈
건설 현장 안전 경고를 Telegram으로 전송

Author: Safety Vision AI Team
Date: 2025-11-23
"""
import os
import requests
from datetime import datetime
from typing import Optional
import io
from PIL import Image


class TelegramNotifier:
    """Telegram Bot을 이용한 알림 전송 클래스"""

    def __init__(self):
        """환경 변수에서 설정 로드"""
        # .env 파일에서 Telegram Bot Token 가져오기
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        # .env 파일에서 채팅방 ID 가져오기
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        # 알림 활성화 여부 확인 (기본값: false)
        self.enabled = os.getenv("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true"

        # 알림이 활성화되었지만 필수 설정이 없는 경우
        if self.enabled and (not self.bot_token or not self.chat_id):
            print("⚠️ Telegram 설정이 없습니다. .env 파일을 확인하세요.")
            self.enabled = False

    def send_safety_alert(
        self,
        head_count: int,
        total_workers: int,
        helmet_rate: float,
        image: Optional[Image.Image] = None,
        location: str = "건설 현장"
    ) -> bool:
        """
        안전 경고 알림 전송

        Args:
            head_count: 헬멧 미착용자 수
            total_workers: 전체 작업자 수
            helmet_rate: 헬멧 착용률 (%)
            image: 탐지 결과 이미지 (PIL Image)
            location: 현장 위치

        Returns:
            bool: 전송 성공 여부
        """
        # 알림이 비활성화된 경우 전송하지 않음
        if not self.enabled:
            return False

        # 안전 수준 판정 (착용률 기준)
        if helmet_rate >= 90:
            # 90% 이상: 우수한 안전 수준
            level = "✅ Excellent"
            emoji = "✅"
            urgency = ""
        elif helmet_rate >= 70:
            # 70~90%: 주의 필요
            level = "⚠️ Caution"
            emoji = "⚠️"
            urgency = "⚠️ 현장 확인이 필요합니다."
        else:
            # 70% 미만: 위험 수준
            level = "🚨 Dangerous"
            emoji = "🚨"
            urgency = "🚨 즉시 확인 필요!"

        # 메시지 작성
        message = f"""{emoji} *Safety Vision AI 경고*

📅 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🏗️ 현장: {location}

⚠️ *헬멧 미착용: {head_count}명*
👷 전체 작업자: {total_workers}명
📊 착용률: {helmet_rate:.1f}%
🛡️ 안전 수준: {level}

{urgency}"""

        try:
            # Telegram Bot API를 통한 텍스트 메시지 전송
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,  # 메시지를 받을 채팅방 ID
                "text": message.strip(),  # 전송할 메시지 내용
                "parse_mode": "Markdown"  # Markdown 형식 지원 (*굵게*, _기울임_ 등)
            }
            # POST 요청으로 메시지 전송 (타임아웃 10초)
            response = requests.post(url, data=data, timeout=10)

            # 텍스트 메시지 전송 성공 시 이미지도 전송
            if image and response.status_code == 200:
                self._send_image(image, helmet_rate)

            # HTTP 200 상태 코드면 성공
            return response.status_code == 200

        except Exception as e:
            # 네트워크 오류, 타임아웃 등 예외 처리
            print(f"❌ Telegram 알림 전송 실패: {e}")
            return False

    def _send_image(self, image: Image.Image, helmet_rate: float):
        """탐지 결과 이미지 전송 (내부 메서드)"""
        try:
            # PIL Image 객체를 바이트 스트림으로 변환
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')  # PNG 형식으로 저장
            img_byte_arr.seek(0)  # 파일 포인터를 처음으로 이동

            # Telegram sendPhoto API 호출
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            files = {'photo': img_byte_arr}  # 이미지 파일 첨부
            data = {
                "chat_id": self.chat_id,  # 메시지를 받을 채팅방 ID
                "caption": f"📸 탐지 결과 (착용률: {helmet_rate:.1f}%)"  # 이미지 설명
            }
            # POST 요청으로 이미지 전송 (타임아웃 10초)
            requests.post(url, data=data, files=files, timeout=10)

        except Exception as e:
            # 이미지 변환 또는 전송 실패 시 경고만 출력 (프로그램은 계속 실행)
            print(f"⚠️ 이미지 전송 실패: {e}")

    def test_connection(self) -> bool:
        """
        Telegram Bot 연결 테스트

        Returns:
            bool: 연결 성공 여부
        """
        # 알림이 비활성화된 경우 테스트 불가
        if not self.enabled:
            return False

        try:
            # Telegram API로 테스트 메시지 전송
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,  # 채팅방 ID
                "text": "✅ Safety Vision AI Bot 연결 테스트 성공!"  # 테스트 메시지
            }
            response = requests.post(url, data=data, timeout=10)
            # HTTP 200 상태 코드면 성공
            return response.status_code == 200
        except:
            # 모든 예외 발생 시 연결 실패로 처리
            return False


# 전역 인스턴스 생성 (모듈 import 시 자동으로 초기화됨)
notifier = TelegramNotifier()
