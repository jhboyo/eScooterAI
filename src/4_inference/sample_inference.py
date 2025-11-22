"""
샘플 이미지 추론 및 시각화 스크립트

이 스크립트는 학습된 YOLOv8 PPE Detection 모델을 사용하여
테스트 데이터셋의 샘플 이미지에 대한 객체 탐지를 수행하고
시각화된 결과를 생성합니다.

주요 기능:
1. YOLOv8 모델 로드
2. 샘플 이미지 선택 및 추론
3. 탐지 결과 시각화 (바운딩 박스, 라벨, 신뢰도)
4. 탐지 통계 계산 및 출력
5. 결과 이미지 저장

사용 방법:
    uv run python src/4_inference/sample_inference.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ============================================================================
# 클래스 정의 및 설정
# ============================================================================

# 탐지 클래스 정보
# - 0: helmet (헬멧 착용) - 안전 ✅
# - 1: head (헬멧 미착용) - 위험 ⚠️
# - 2: vest (안전조끼) - 안전 ✅
CLASS_NAMES = {
    0: 'helmet',  # 헬멧을 착용한 머리
    1: 'head',    # 헬멧 없는 머리 (안전 위반!)
    2: 'vest'     # 안전 조끼
}

# 클래스별 시각화 색상 (RGB 형식)
# 색상 선택 이유:
# - 파란색(helmet): 안전한 상태를 나타내는 차분한 색
# - 빨간색(head): 위험 상태를 나타내는 경고색
# - 노란색(vest): 안전 장비를 나타내는 시인성 높은 색
CLASS_COLORS = {
    0: (0, 0, 255),     # helmet - 파란색 (안전)
    1: (255, 0, 0),     # head - 빨간색 (위험!)
    2: (255, 255, 0)    # vest - 노란색 (안전장비)
}

# ============================================================================
# 탐지 및 시각화 함수
# ============================================================================

def detect_and_visualize(model, image_path, output_dir):
    """
    단일 이미지에 대한 객체 탐지 수행 및 결과 시각화

    이 함수는 다음 작업을 수행합니다:
    1. 이미지 파일 읽기 및 전처리
    2. YOLOv8 모델을 사용한 객체 탐지
    3. 원본과 탐지 결과를 나란히 비교하는 시각화
    4. 탐지 통계 계산
    5. 결과 이미지 저장

    Args:
        model: 로드된 YOLOv8 모델 객체
        image_path: 처리할 이미지 파일 경로 (Path 객체)
        output_dir: 결과를 저장할 디렉토리 경로 (Path 객체)

    Returns:
        list: 탐지된 객체 정보 리스트 (클래스, 신뢰도, 바운딩 박스)
              각 항목은 {'class': str, 'confidence': float, 'bbox': list} 형식
    """

    # ========================================
    # 1. 이미지 읽기 및 전처리
    # ========================================
    # OpenCV를 사용하여 이미지 읽기 (BGR 형식으로 읽힘)
    image = cv2.imread(str(image_path))

    # 이미지 읽기 실패 처리
    if image is None:
        print(f"⚠️ 이미지를 읽을 수 없습니다: {image_path}")
        return []

    # matplotlib 시각화를 위해 BGR → RGB 변환
    # OpenCV는 BGR, matplotlib는 RGB 형식을 사용
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ========================================
    # 2. YOLOv8 객체 탐지 수행
    # ========================================
    # conf=0.25: 신뢰도 25% 이상인 탐지만 유지
    # 너무 낮으면 오탐지 증가, 너무 높으면 미탐지 증가
    results = model(image, conf=0.25)

    # ========================================
    # 3. 시각화 준비
    # ========================================
    # 1x2 subplot 생성 (원본 vs 탐지 결과)
    # figsize=(16, 8): 충분한 크기로 설정하여 디테일 확인 가능
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 왼쪽: 원본 이미지
    ax1.imshow(image_rgb)
    ax1.set_title(f'Original: {image_path.name}', fontsize=12)
    ax1.axis('off')  # 축 제거로 깔끔한 표시

    # 오른쪽: 탐지 결과
    ax2.imshow(image_rgb)
    ax2.set_title('Detection Results', fontsize=12)
    ax2.axis('off')

    # ========================================
    # 4. 탐지된 객체 처리 및 시각화
    # ========================================
    # 탐지 정보를 저장할 리스트
    detection_info = []

    # 각 탐지 결과 처리
    for r in results:
        boxes = r.boxes  # 탐지된 바운딩 박스들

        if boxes is not None:
            # 각 바운딩 박스 처리
            for box in boxes:
                # --------------------------------
                # 4.1 탐지 정보 추출
                # --------------------------------
                # xyxy: [x1, y1, x2, y2] 형식의 좌표
                # x1, y1: 좌상단 좌표
                # x2, y2: 우하단 좌표
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # 신뢰도 점수 (0~1 범위)
                conf = box.conf[0].cpu().numpy()

                # 클래스 ID (0: helmet, 1: head, 2: vest)
                cls = int(box.cls[0].cpu().numpy())

                # --------------------------------
                # 4.2 클래스 정보 매핑
                # --------------------------------
                # 클래스 ID를 이름으로 변환
                class_name = CLASS_NAMES.get(cls, f'class_{cls}')

                # 클래스별 색상 지정 (기본값: 회색)
                color = CLASS_COLORS.get(cls, (128, 128, 128))

                # --------------------------------
                # 4.3 바운딩 박스 시각화
                # --------------------------------
                # matplotlib Rectangle 객체 생성
                # - (x1, y1): 시작점
                # - width: x2 - x1
                # - height: y2 - y1
                rect = patches.Rectangle(
                    (x1, y1),                           # 시작점
                    x2 - x1,                           # 너비
                    y2 - y1,                           # 높이
                    linewidth=2,                       # 선 두께
                    edgecolor=np.array(color)/255,     # 테두리 색 (0~1로 정규화)
                    facecolor='none'                   # 내부 투명
                )
                ax2.add_patch(rect)

                # --------------------------------
                # 4.4 라벨 표시
                # --------------------------------
                # 클래스명과 신뢰도를 함께 표시
                label = f'{class_name}: {conf:.2f}'  # 예: "helmet: 0.92"

                # 바운딩 박스 위에 라벨 추가
                # y1 - 5: 박스 위쪽에 약간 떨어뜨려 표시
                ax2.text(
                    x1, y1 - 5,                        # 위치
                    label,                             # 텍스트
                    color=np.array(color)/255,        # 글자색
                    fontsize=10,                       # 글자 크기
                    bbox=dict(                         # 배경 박스
                        boxstyle='round,pad=0.3',     # 둥근 모서리
                        facecolor='white',             # 흰 배경
                        alpha=0.7                      # 반투명
                    )
                )

                # --------------------------------
                # 4.5 탐지 정보 저장
                # --------------------------------
                detection_info.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })

    # ========================================
    # 5. 탐지 통계 계산
    # ========================================
    # 각 클래스별 탐지 개수 집계
    helmet_count = sum(1 for d in detection_info if d['class'] == 'helmet')
    head_count = sum(1 for d in detection_info if d['class'] == 'head')
    vest_count = sum(1 for d in detection_info if d['class'] == 'vest')

    # 전체 그림 상단에 통계 표시
    stats_text = f'Detections: Helmet={helmet_count}, Head={head_count}, Vest={vest_count}'
    fig.suptitle(stats_text, fontsize=14, fontweight='bold')

    # ========================================
    # 6. 결과 저장
    # ========================================
    # 출력 파일명 생성 (detection_원본파일명.png)
    output_path = output_dir / f'detection_{image_path.stem}.png'

    # 그림 저장 (dpi=100: 적절한 해상도)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()  # 메모리 해제를 위해 figure 닫기

    # ========================================
    # 7. 처리 결과 출력
    # ========================================
    print(f"✅ 처리 완료: {image_path.name}")
    print(f"   - 헬멧 착용: {helmet_count}명")
    print(f"   - 헬멧 미착용: {head_count}명 {'⚠️' if head_count > 0 else ''}")
    print(f"   - 안전조끼: {vest_count}개")
    print(f"   - 저장 위치: {output_path}")
    print()

    return detection_info

# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """
    메인 실행 함수

    전체 실행 흐름:
    1. 환경 설정 (경로, 디렉토리)
    2. YOLOv8 모델 로드
    3. 샘플 이미지 선택
    4. 각 이미지에 대한 탐지 및 시각화
    5. 전체 통계 계산 및 출력
    """

    # ========================================
    # 1. 실행 시작 알림
    # ========================================
    print("="*70)
    print("PPE Detection 샘플 추론 시연")
    print("="*70)
    print()

    # ========================================
    # 2. 경로 설정
    # ========================================
    # 현재 스크립트 기준 프로젝트 루트 경로 계산
    # __file__: 현재 스크립트 파일 경로
    # .parent.parent.parent: 3단계 상위 (src/4_inference/sample_inference.py → 루트)
    base_dir = Path(__file__).parent.parent.parent

    # 모델 파일 경로 (훈련된 best.pt)
    model_path = base_dir / 'models' / 'ppe_detection' / 'weights' / 'best.pt'

    # 테스트 이미지 디렉토리
    test_dir = base_dir / 'dataset' / 'data' / 'test' / 'images'

    # 출력 디렉토리 (결과 저장)
    output_dir = base_dir / 'output' / 'sample_detections'

    # 출력 디렉토리가 없으면 생성
    # parents=True: 상위 디렉토리도 함께 생성
    # exist_ok=True: 이미 존재해도 에러 발생하지 않음
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================
    # 3. YOLOv8 모델 로드
    # ========================================
    print("🤖 모델 로드 중...")
    model = YOLO(str(model_path))
    print("   ✅ 모델 로드 완료")
    print()

    # ========================================
    # 4. 샘플 이미지 선택
    # ========================================
    # 다양한 시나리오를 포함하도록 샘플 선정:
    # - ds1: 헬멧 중심 데이터셋
    # - ds2: 헬멧 + 조끼 데이터셋
    # - 그룹샷, 개인샷 등 다양한 구도
    sample_images = [
        'ds1_hard_hat_workers0.png',    # 그룹 샷 (여러 작업자)
        'ds1_hard_hat_workers10.png',   # 소규모 그룹
        'ds1_hard_hat_workers50.png',   # 개인 또는 소수
        'ds2_helmet_jacket_00003.jpg',  # 헬멧 + 조끼 (개인)
        'ds2_helmet_jacket_00023.jpg'   # 헬멧 + 조끼 (그룹)
    ]

    print(f"📸 {len(sample_images)}개 샘플 이미지 처리 중...")
    print("-"*70)

    # ========================================
    # 5. 각 이미지 처리
    # ========================================
    # 모든 탐지 결과를 저장할 리스트
    all_detections = []

    # 각 샘플 이미지 순차 처리
    for img_name in sample_images:
        # 이미지 전체 경로 생성
        img_path = test_dir / img_name

        # 파일 존재 확인
        if img_path.exists():
            # 탐지 및 시각화 수행
            detections = detect_and_visualize(model, img_path, output_dir)

            # 결과 저장
            all_detections.append({
                'image': img_name,
                'detections': detections
            })
        else:
            print(f"⚠️ 이미지를 찾을 수 없습니다: {img_name}")

    # ========================================
    # 6. 전체 처리 완료 알림
    # ========================================
    print("="*70)
    print(f"✅ 모든 처리 완료!")
    print(f"📂 결과 저장 위치: {output_dir}")
    print("="*70)

    # ========================================
    # 7. 전체 통계 계산 및 출력
    # ========================================
    # 모든 이미지에서 탐지된 객체 총계 계산
    # 중첩된 리스트 컴프리헨션으로 전체 합계 계산

    # 총 헬멧 착용자 수
    total_helmet = sum(
        sum(1 for d in result['detections'] if d['class'] == 'helmet')
        for result in all_detections if result['detections']
    )

    # 총 헬멧 미착용자 수
    total_head = sum(
        sum(1 for d in result['detections'] if d['class'] == 'head')
        for result in all_detections if result['detections']
    )

    # 총 안전조끼 수
    total_vest = sum(
        sum(1 for d in result['detections'] if d['class'] == 'vest')
        for result in all_detections if result['detections']
    )

    # 통계 출력
    print("\n📊 전체 탐지 통계:")
    print(f"   - 총 헬멧 착용: {total_helmet}명")
    print(f"   - 총 헬멧 미착용: {total_head}명")
    print(f"   - 총 안전조끼: {total_vest}개")

    # 헬멧 착용률 계산 (전체 작업자가 있는 경우만)
    if total_helmet + total_head > 0:
        helmet_rate = total_helmet / (total_helmet + total_head) * 100
        print(f"   - 헬멧 착용률: {helmet_rate:.1f}%")

        # 안전 수준 평가
        if helmet_rate >= 90:
            print("   - 안전 수준: ✅ 우수")
        elif helmet_rate >= 70:
            print("   - 안전 수준: ⚠️ 주의 필요")
        else:
            print("   - 안전 수준: 🚨 위험")

# ============================================================================
# 스크립트 직접 실행 시
# ============================================================================
if __name__ == '__main__':
    # 이 스크립트가 직접 실행될 때만 main() 함수 호출
    # 모듈로 import되는 경우에는 실행되지 않음
    main()