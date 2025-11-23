# Streamlit Cloud Secrets 설정 가이드

Streamlit Cloud에서 Telegram Bot 알림 기능을 사용하려면 환경 변수를 Secrets로 설정해야 합니다.

## 설정 방법

### 1. Streamlit Cloud 접속
- https://share.streamlit.io 접속
- 본인의 앱 선택

### 2. Secrets 설정 메뉴 진입
- 앱 대시보드에서 **Settings** 클릭
- **Secrets** 탭 선택

### 3. Secrets 추가 (TOML 형식)

아래 내용을 복사하여 Secrets 입력창에 붙여넣기:

```toml
# Telegram Bot 알림 설정
TELEGRAM_BOT_TOKEN = "8572351389:AAHWObTgDzEGXrVcO5K6RAf2pydDzDIgJUQ"
TELEGRAM_CHAT_ID = "1955008331"
TELEGRAM_ALERTS_ENABLED = "true"
```

### 4. Save 버튼 클릭

### 5. 앱 재시작
- Secrets 설정 후 앱이 자동으로 재시작됩니다.
- 사이드바의 "📱 알림 설정"에서 "✅ Telegram 알림 활성화" 확인

## 주의사항

⚠️ **보안 주의**
- Bot Token은 절대 공개 저장소에 커밋하지 마세요
- `.env` 파일은 `.gitignore`에 포함되어야 합니다
- Streamlit Cloud Secrets는 암호화되어 안전하게 저장됩니다

## 테스트

Streamlit 앱 실행 후:
1. 사이드바에서 "🔔 연결 테스트" 버튼 클릭
2. "✅ Telegram 연결 성공!" 메시지 확인
3. Telegram 앱에서 테스트 메시지 수신 확인

## 문제 해결

### "Telegram 알림이 비활성화되어 있습니다" 메시지가 표시되는 경우
- Secrets 설정 확인 (TELEGRAM_ALERTS_ENABLED = "true")
- 앱 재시작

### "Telegram 연결 실패" 메시지가 표시되는 경우
- TELEGRAM_BOT_TOKEN 확인
- TELEGRAM_CHAT_ID 확인
- Bot과 대화 시작 여부 확인 (/start 전송)
