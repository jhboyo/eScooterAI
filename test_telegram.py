"""
텔레그램 알림 기능 테스트 스크립트
"""
import sys
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
load_dotenv(project_root / ".env")

from src.alert import notifier

def create_dummy_image():
    """테스트용 더미 이미지 생성"""
    # 640x480 크기의 빈 이미지 생성
    img = Image.new('RGB', (640, 480), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # 텍스트 추가
    text = "eScooter AI Test Image"
    # 기본 폰트 사용
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 40)
    except:
        font = ImageFont.load_default()

    # 중앙에 텍스트 그리기
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((640 - text_width) // 2, (480 - text_height) // 2)

    draw.text(position, text, fill=(59, 130, 246), font=font)

    return img

def main():
    """텔레그램 알림 기능 테스트"""
    print("=" * 60)
    print("🤖 eScooter AI - Telegram Bot 기능 점검")
    print("=" * 60)
    print()

    # 1. 설정 확인
    print("📋 1단계: 설정 확인")
    print(f"   - Telegram 알림 활성화: {notifier.enabled}")
    print(f"   - Bot Token 설정: {'✅' if notifier.bot_token else '❌'}")
    print(f"   - Chat ID 설정: {'✅' if notifier.chat_id else '❌'}")
    print()

    if not notifier.enabled:
        print("❌ Telegram 알림이 비활성화되어 있습니다.")
        print("   .env 파일에서 TELEGRAM_ALERTS_ENABLED=true로 설정하세요.")
        return

    # 2. 연결 테스트
    print("🔗 2단계: Telegram Bot 연결 테스트")
    test_result = notifier.test_connection()
    if test_result:
        print("   ✅ 연결 테스트 성공! Telegram 앱에서 메시지를 확인하세요.")
    else:
        print("   ❌ 연결 테스트 실패! Bot Token과 Chat ID를 확인하세요.")
        return
    print()

    # 3. 안전 알림 테스트 (텍스트만)
    print("📝 3단계: 안전 알림 테스트 (텍스트)")
    alert_result = notifier.send_safety_alert(
        head_count=2,
        total_workers=10,
        helmet_rate=80.0,
        location="테스트 구역"
    )
    if alert_result:
        print("   ✅ 텍스트 알림 전송 성공!")
    else:
        print("   ❌ 텍스트 알림 전송 실패!")
    print()

    # 4. 이미지 포함 알림 테스트
    print("📸 4단계: 이미지 포함 알림 테스트")
    dummy_image = create_dummy_image()
    image_result = notifier.send_safety_alert(
        head_count=3,
        total_workers=15,
        helmet_rate=80.0,
        image=dummy_image,
        location="테스트 구역 (이미지 포함)"
    )
    if image_result:
        print("   ✅ 이미지 포함 알림 전송 성공!")
    else:
        print("   ❌ 이미지 포함 알림 전송 실패!")
    print()

    # 5. 위험 수준별 테스트
    print("🚨 5단계: 위험 수준별 알림 테스트")

    print("   5-1. 안전 수준 (90% 이상)")
    notifier.send_safety_alert(
        head_count=1,
        total_workers=10,
        helmet_rate=90.0,
        location="안전 구역"
    )
    print("   ✅ 안전 수준 알림 전송")

    print("   5-2. 주의 수준 (70~90%)")
    notifier.send_safety_alert(
        head_count=3,
        total_workers=10,
        helmet_rate=70.0,
        location="주의 구역"
    )
    print("   ✅ 주의 수준 알림 전송")

    print("   5-3. 위험 수준 (70% 미만)")
    notifier.send_safety_alert(
        head_count=5,
        total_workers=10,
        helmet_rate=50.0,
        location="위험 구역"
    )
    print("   ✅ 위험 수준 알림 전송")
    print()

    # 완료
    print("=" * 60)
    print("✅ 텔레그램 알림 기능 점검 완료!")
    print("   Telegram 앱에서 모든 메시지를 확인하세요.")
    print("=" * 60)

if __name__ == "__main__":
    main()
