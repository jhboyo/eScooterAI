"""
Telegram Bot 알림 기능 테스트 스크립트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# .env 파일 로드 (IMPORTANT: import 전에 로드해야 함!)
from dotenv import load_dotenv
load_dotenv()

# Alert 모듈 import
from src.alert.telegram_notifier import notifier
from PIL import Image
import os

def test_connection():
    """연결 테스트"""
    print("=" * 60)
    print("1️⃣ Telegram Bot 연결 테스트")
    print("=" * 60)

    print(f"Bot Token: {notifier.bot_token[:20]}..." if notifier.bot_token else "Bot Token: None")
    print(f"Chat ID: {notifier.chat_id}")
    print(f"Enabled: {notifier.enabled}")
    print()

    if notifier.test_connection():
        print("✅ 연결 성공! Telegram에서 메시지를 확인하세요.")
        return True
    else:
        print("❌ 연결 실패. Bot Token과 Chat ID를 확인하세요.")
        return False

def test_safety_alert():
    """안전 경고 알림 테스트"""
    print("\n" + "=" * 60)
    print("2️⃣ 안전 경고 알림 테스트")
    print("=" * 60)

    # 테스트 데이터
    test_cases = [
        {
            "name": "위험 상황 (헬멧 착용률 50%)",
            "head_count": 5,
            "total_workers": 10,
            "helmet_rate": 50.0
        },
        {
            "name": "주의 상황 (헬멧 착용률 80%)",
            "head_count": 2,
            "total_workers": 10,
            "helmet_rate": 80.0
        },
        {
            "name": "안전 상황 (헬멧 착용률 95%)",
            "head_count": 0,
            "total_workers": 20,
            "helmet_rate": 95.0
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n[테스트 {i}] {case['name']}")
        success = notifier.send_safety_alert(
            head_count=case['head_count'],
            total_workers=case['total_workers'],
            helmet_rate=case['helmet_rate'],
            image=None,  # 이미지 없이 텍스트만 전송
            location=f"테스트 현장 #{i}"
        )

        if success:
            print(f"✅ 알림 전송 성공")
        else:
            print(f"❌ 알림 전송 실패")

        # 다음 테스트 전 대기
        if i < len(test_cases):
            import time
            print("   (2초 대기...)")
            time.sleep(2)

def test_with_image():
    """이미지 포함 알림 테스트"""
    print("\n" + "=" * 60)
    print("3️⃣ 이미지 포함 알림 테스트")
    print("=" * 60)

    # 테스트 이미지 생성 (100x100 빨간색 이미지)
    test_image = Image.new('RGB', (200, 200), color='red')

    # 이미지에 텍스트 추가 (선택사항)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_image)
    draw.text((50, 90), "TEST IMAGE", fill='white')

    success = notifier.send_safety_alert(
        head_count=3,
        total_workers=8,
        helmet_rate=62.5,
        image=test_image,
        location="테스트 현장 (이미지 포함)"
    )

    if success:
        print("✅ 이미지 포함 알림 전송 성공")
    else:
        print("❌ 이미지 포함 알림 전송 실패")

if __name__ == "__main__":
    print("\n🤖 Telegram Bot 알림 기능 테스트 시작\n")

    # 1. 연결 테스트
    if not test_connection():
        print("\n❌ 연결 테스트 실패. 환경 변수를 확인하세요.")
        sys.exit(1)

    # 2. 안전 경고 알림 테스트
    test_safety_alert()

    # 3. 이미지 포함 알림 테스트
    test_with_image()

    print("\n" + "=" * 60)
    print("✅ 모든 테스트 완료!")
    print("=" * 60)
    print("\n📱 Telegram 앱에서 알림을 확인하세요.\n")
