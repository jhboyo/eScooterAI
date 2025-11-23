# Telegram Bot 안전 경고 알림 구현 계획서

> Safety Vision AI - 실시간 안전 경고 알림 시스템 추가

**작성일**: 2025-11-23
**예상 소요 시간**: 2-3시간
**난이도**: ⭐⭐☆☆☆ (쉬움)

---

## 📋 목차

1. [개요](#개요)
2. [기능 명세](#기능-명세)
3. [구현 단계](#구현-단계)
4. [파일 구조](#파일-구조)
5. [테스트 시나리오](#테스트-시나리오)

---

## 🎯 개요

### 목적
건설 현장에서 헬멧 미착용자 발견 시 관리자 핸드폰으로 **즉시 알림**을 전송하여 신속한 대응 가능

### 주요 기능
- ✅ 헬멧 미착용자(head) 탐지 시 자동 알림
- ✅ 안전 수준(Dangerous) 도달 시 경고 알림
- ✅ 탐지 결과 이미지 전송
- ✅ 실시간 통계 정보 제공

### 선택한 플랫폼: Telegram
- ✅ **무료** (API 무료, 사용량 제한 없음)
- ✅ **간단한 설정** (10-15분)
- ✅ **실시간 푸시 알림**
- ✅ **이미지 전송 가능**
- ✅ **크로스 플랫폼** (iOS, Android, Desktop)

---

## 🎨 기능 명세

### 알림 트리거 조건

| 조건 | 알림 발송 |
|------|----------|
| Head 클래스 1개 이상 탐지 | ✅ 경고 알림 |
| 헬멧 착용률 < 70% (Dangerous) | ✅ 위험 알림 |
| 헬멧 착용률 70-89% (Caution) | ⚠️ 주의 알림 (선택사항) |
| 헬멧 착용률 ≥ 90% (Excellent) | ℹ️ 정상 알림 (선택사항) |

### 알림 메시지 형식

```
🚨 Safety Vision AI 경고

📅 시간: 2025-11-23 15:30:42
🏗️ 현장: 건설 현장 A동

⚠️ 헬멧 미착용: 3명
👷 전체 작업자: 12명
📊 착용률: 75.0%
🛡️ 안전 수준: Caution

[탐지 결과 이미지]

⚠️ 현장 확인이 필요합니다.
```

---

## 🔧 구현 단계

### Phase 1: Telegram Bot 설정 (10-15분)

#### Step 1-1: Bot 생성

1. **Telegram 앱 실행**
2. **BotFather 검색** (`@BotFather`)
3. `/newbot` 명령 입력
4. Bot 이름 입력: `Safety Vision AI Bot`
5. Bot username 입력: `safetyvisionai_bot` (또는 원하는 이름)
6. **Bot Token 저장** (예: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

#### Step 1-2: Chat ID 획득

1. Bot과 대화 시작 (`/start` 입력)
2. 다음 URL 접속:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
3. JSON 응답에서 `chat.id` 확인 (예: `987654321`)

#### Step 1-3: 환경 변수 설정

**파일: `.env` (루트 디렉토리)**
```bash
# Telegram Bot 설정
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321
TELEGRAM_ALERTS_ENABLED=true
```

---

### Phase 2: 알림 모듈 개발 (30-45분)

#### Step 2-1: 의존성 추가

**파일: `requirements.txt`**
```txt
# 기존 의존성...
requests>=2.31.0  # HTTP 요청 (Telegram API 호출용)
```

#### Step 2-2: 알림 모듈 작성

**파일: `src/5_web_interface/utils/telegram_notifier.py` (새로 생성)**

```python
"""
Telegram Bot 알림 모듈
건설 현장 안전 경고를 Telegram으로 전송
"""
import os
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import io
from PIL import Image


class TelegramNotifier:
    """Telegram Bot을 이용한 알림 전송 클래스"""

    def __init__(self):
        """환경 변수에서 설정 로드"""
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = os.getenv("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true"

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
        if not self.enabled:
            return False

        # 안전 수준 판정
        if helmet_rate >= 90:
            level = "✅ Excellent"
            emoji = "✅"
            urgency = ""
        elif helmet_rate >= 70:
            level = "⚠️ Caution"
            emoji = "⚠️"
            urgency = "⚠️ 현장 확인이 필요합니다."
        else:
            level = "🚨 Dangerous"
            emoji = "🚨"
            urgency = "🚨 즉시 확인 필요!"

        # 메시지 작성
        message = f"""
{emoji} *Safety Vision AI 경고*

📅 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🏗️ 현장: {location}

⚠️ *헬멧 미착용: {head_count}명*
👷 전체 작업자: {total_workers}명
📊 착용률: {helmet_rate:.1f}%
🛡️ 안전 수준: {level}

{urgency}
        """

        try:
            # 텍스트 메시지 전송
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message.strip(),
                "parse_mode": "Markdown"
            }
            response = requests.post(url, data=data, timeout=10)

            # 이미지 전송
            if image and response.status_code == 200:
                self._send_image(image, helmet_rate)

            return response.status_code == 200

        except Exception as e:
            print(f"❌ Telegram 알림 전송 실패: {e}")
            return False

    def _send_image(self, image: Image.Image, helmet_rate: float):
        """탐지 결과 이미지 전송"""
        try:
            # PIL Image를 bytes로 변환
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            files = {'photo': img_byte_arr}
            data = {
                "chat_id": self.chat_id,
                "caption": f"📸 탐지 결과 (착용률: {helmet_rate:.1f}%)"
            }
            requests.post(url, data=data, files=files, timeout=10)

        except Exception as e:
            print(f"⚠️ 이미지 전송 실패: {e}")

    def test_connection(self) -> bool:
        """Telegram Bot 연결 테스트"""
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": "✅ Safety Vision AI Bot 연결 테스트 성공!"
            }
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
        except:
            return False


# 전역 인스턴스
notifier = TelegramNotifier()
```

---

### Phase 3: Streamlit 앱 통합 (30-45분)

#### Step 3-1: app.py에 알림 기능 추가

**파일: `src/5_web_interface/app.py`**

1. **Import 추가** (파일 상단)
```python
from utils.telegram_notifier import notifier
```

2. **사이드바에 Telegram 설정 추가** (sidebar_config 함수 내)
```python
# 기존 사이드바 코드 아래에 추가
st.markdown("---")
st.header("📱 알림 설정")

# Telegram 알림 상태 표시
if notifier.enabled:
    st.success("✅ Telegram 알림 활성화")
    if st.button("🔔 연결 테스트"):
        if notifier.test_connection():
            st.success("✅ Telegram 연결 성공!")
        else:
            st.error("❌ Telegram 연결 실패")
else:
    st.info("ℹ️ Telegram 알림 비활성화")
    st.caption("`.env` 파일에서 설정 가능")
```

3. **탐지 완료 후 알림 전송** (탐지 결과 처리 부분)
```python
# 추론 완료 후 (line ~290 근처)
if results:
    # 기존 코드...

    # Telegram 알림 전송 (head가 1개 이상이거나 착용률이 낮을 때)
    if head_total > 0 or helmet_rate < 90:
        # 첫 번째 이미지의 탐지 결과 사용
        result_image = results[0].get('annotated_image') if results else None

        success = notifier.send_safety_alert(
            head_count=head_total,
            total_workers=person_total,
            helmet_rate=helmet_rate,
            image=result_image,
            location="건설 현장 A동"
        )

        if success:
            st.info("📱 Telegram 알림이 전송되었습니다!")
```

---

### Phase 4: 환경 설정 (5-10분)

#### Step 4-1: .env 파일 생성/수정

**파일: `.env` (루트 디렉토리)**
```bash
# 기존 설정...
PROJECT_ROOT=/Users/joonho/workspace/sogang/tf-basic/SafetyVisionAI

# Telegram Bot 설정 (추가)
TELEGRAM_BOT_TOKEN=<YOUR_BOT_TOKEN>
TELEGRAM_CHAT_ID=<YOUR_CHAT_ID>
TELEGRAM_ALERTS_ENABLED=true
```

#### Step 4-2: .env.example 업데이트

**파일: `.env.example`**
```bash
PROJECT_ROOT=/path/to/SafetyVisionAI

# Telegram Bot 설정 (선택사항)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELEGRAM_ALERTS_ENABLED=false
```

---

### Phase 5: 배포 준비 (10-15분)

#### Streamlit Cloud에서 Secrets 설정

**Streamlit Cloud 대시보드:**
1. 앱 선택 → Settings → Secrets
2. 다음 내용 추가:

```toml
# Telegram Bot Secrets
TELEGRAM_BOT_TOKEN = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
TELEGRAM_CHAT_ID = "987654321"
TELEGRAM_ALERTS_ENABLED = "true"
```

**주의:** `.env` 파일은 `.gitignore`에 포함되어 GitHub에 올라가지 않으므로, Streamlit Cloud에서 별도로 Secrets 설정 필요

---

## 📁 파일 구조

```
SafetyVisionAI/
├── .env                                # Telegram Bot 설정 (gitignore)
├── .env.example                        # 설정 예시
├── requirements.txt                    # requests 추가
├── src/
│   └── 5_web_interface/
│       ├── app.py                      # Telegram 통합
│       └── utils/
│           ├── inference.py
│           ├── plotting.py
│           └── telegram_notifier.py    # 🆕 새로 생성
└── TELEGRAM_ALERT_PLAN.md             # 이 문서
```

---

## 🧪 테스트 시나리오

### Test 1: 연결 테스트
1. Streamlit 앱 실행
2. 사이드바 "연결 테스트" 버튼 클릭
3. Telegram 앱에서 메시지 수신 확인

**예상 결과:**
```
✅ Safety Vision AI Bot 연결 테스트 성공!
```

---

### Test 2: Head 탐지 알림
1. 헬멧 미착용 이미지 업로드
2. "탐지 시작" 버튼 클릭
3. Telegram 앱 확인

**예상 결과:**
```
🚨 Safety Vision AI 경고

📅 시간: 2025-11-23 15:30:42
🏗️ 현장: 건설 현장 A동

⚠️ 헬멧 미착용: 2명
👷 전체 작업자: 5명
📊 착용률: 60.0%
🛡️ 안전 수준: 🚨 Dangerous

🚨 즉시 확인 필요!

[탐지 결과 이미지]
```

---

### Test 3: 안전한 상황 (알림 없음)
1. 모두 헬멧 착용한 이미지 업로드
2. "탐지 시작" 버튼 클릭
3. Telegram 알림 없음 (조건: head > 0 또는 helmet_rate < 90)

**예상 결과:**
- Telegram 알림 없음 (정상)
- 웹 UI에만 "✅ Excellent" 표시

---

## ⏱️ 타임라인

| Phase | 작업 | 소요 시간 |
|-------|------|----------|
| Phase 1 | Telegram Bot 설정 | 10-15분 |
| Phase 2 | 알림 모듈 개발 | 30-45분 |
| Phase 3 | Streamlit 앱 통합 | 30-45분 |
| Phase 4 | 환경 설정 | 5-10분 |
| Phase 5 | 테스트 및 배포 | 15-20분 |
| **합계** | | **1.5~2.5시간** |

---

## ✅ 체크리스트

### 개발 전
- [ ] Telegram 앱 설치 (iOS/Android)
- [ ] BotFather에서 Bot 생성
- [ ] Bot Token 및 Chat ID 확보
- [ ] `.env` 파일에 설정 추가

### 개발 중
- [ ] `telegram_notifier.py` 모듈 작성
- [ ] `requirements.txt`에 `requests` 추가
- [ ] `app.py`에 알림 기능 통합
- [ ] 연결 테스트 버튼 추가

### 테스트
- [ ] 로컬에서 연결 테스트 성공
- [ ] Head 탐지 시 알림 전송 확인
- [ ] 이미지 전송 확인
- [ ] 다양한 착용률에서 테스트

### 배포
- [ ] Streamlit Cloud Secrets 설정
- [ ] GitHub에 푸시 (.env 제외)
- [ ] 배포 후 알림 테스트

---

## 🎯 발표 시연 시나리오

### 데모 흐름
1. **상황 설명**: "건설 현장 관리자가 사무실에 있는 상황"
2. **이미지 업로드**: 헬멧 미착용 이미지 업로드
3. **탐지 시작**: 버튼 클릭
4. **결과 확인**: 웹 UI에 탐지 결과 표시
5. **핸드폰 확인**: 🎬 **Telegram 앱에서 실시간 알림 수신!**
6. **강조**: "관리자가 즉시 현장에 조치 가능"

**임팩트**:
- ✅ 실시간 알림의 실용성 강조
- ✅ 핸드폰 알림으로 시각적 효과
- ✅ 즉각적인 안전 대응 가능

---

## 📚 참고 자료

### Telegram Bot API
- [Official Bot API](https://core.telegram.org/bots/api)
- [BotFather Guide](https://core.telegram.org/bots#botfather)

### 유사 사례
- 스마트 공장 이상 감지 알림 시스템
- 보안 CCTV 침입 감지 알림
- IoT 센서 임계값 초과 알림

---

## 🔜 향후 확장 가능성

### 추가 기능 아이디어
- [ ] 여러 관리자에게 동시 알림 (그룹 채팅)
- [ ] 시간대별 알림 필터 (야간 작업 시에만)
- [ ] 주간/월간 안전 리포트 자동 전송
- [ ] 알림 히스토리 데이터베이스 저장
- [ ] Slack, Discord 등 다른 플랫폼 지원

---

**작성자**: Safety Vision AI Team
**최종 수정**: 2025-11-23
