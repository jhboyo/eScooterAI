"""
Head 클래스(헬멧 미착용) 탐지 데모 스크립트

헬멧 미착용 작업자를 탐지하여 안전 경고를 시연합니다.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 클래스 정보
CLASS_NAMES = {
    0: 'helmet',
    1: 'head',
    2: 'vest'
}

# 클래스별 색상 (RGB 형식)
CLASS_COLORS = {
    0: (0, 0, 255),     # helmet - 파란색 (안전)
    1: (255, 0, 0),     # head - 빨간색 (위험!)
    2: (255, 255, 0)    # vest - 노란색
}

def detect_and_warn(model, image_path, output_dir):
    """
    단일 이미지에서 헬멧 미착용자 탐지 및 경고
    """
    # 이미지 읽기
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"⚠️ 이미지를 읽을 수 없습니다: {image_path}")
        return []

    # RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 탐지 수행
    results = model(image, conf=0.25)

    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 원본 이미지
    ax1.imshow(image_rgb)
    ax1.set_title(f'Original: {image_path.name}')
    ax1.axis('off')

    # 탐지 결과
    ax2.imshow(image_rgb)
    ax2.set_title('Safety Violation Detection')
    ax2.axis('off')

    # 탐지 정보
    detection_info = []
    head_locations = []  # 헬멧 미착용자 위치

    # 바운딩 박스 그리기
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                class_name = CLASS_NAMES.get(cls, f'class_{cls}')
                color = CLASS_COLORS.get(cls, (128, 128, 128))

                # 바운딩 박스
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=3 if cls == 1 else 2,  # head는 두껍게
                    edgecolor=np.array(color)/255,
                    facecolor='none'
                )
                ax2.add_patch(rect)

                # 라벨
                label = f'{class_name}: {conf:.2f}'
                if cls == 1:  # head 클래스인 경우
                    label = f'⚠️ {label}'
                    head_locations.append((x1 + (x2-x1)/2, y1 + (y2-y1)/2))

                ax2.text(x1, y1 - 5, label,
                        color=np.array(color)/255, fontsize=11,
                        fontweight='bold' if cls == 1 else 'normal',
                        bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='yellow' if cls == 1 else 'white',
                                alpha=0.9 if cls == 1 else 0.7))

                detection_info.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                })

    # 경고 화살표 추가
    for x, y in head_locations:
        ax2.annotate('DANGER!', xy=(x, y),
                    xytext=(x, y-50),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, color='red', fontweight='bold',
                    ha='center')

    # 통계 계산
    helmet_count = sum(1 for d in detection_info if d['class'] == 'helmet')
    head_count = sum(1 for d in detection_info if d['class'] == 'head')
    vest_count = sum(1 for d in detection_info if d['class'] == 'vest')

    total_workers = helmet_count + head_count

    # 안전 상태 판단
    if total_workers > 0:
        helmet_rate = helmet_count / total_workers * 100
        if head_count == 0:
            status = "✅ SAFE"
            status_color = 'green'
        elif helmet_rate >= 70:
            status = "⚠️ CAUTION"
            status_color = 'orange'
        else:
            status = "🚨 DANGER"
            status_color = 'red'
    else:
        status = "NO WORKERS"
        status_color = 'gray'
        helmet_rate = 0

    # 제목 업데이트
    stats_text = (f'Status: {status} | '
                 f'Helmet={helmet_count}, Head={head_count}, '
                 f'Compliance={helmet_rate:.1f}%')
    fig.suptitle(stats_text, fontsize=14, fontweight='bold', color=status_color)

    # 저장
    output_path = output_dir / f'safety_{image_path.stem}.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    # 콘솔 출력
    print(f"\n{'='*60}")
    print(f"📸 이미지: {image_path.name}")
    print(f"{'='*60}")
    print(f"👷 작업자 현황:")
    print(f"   - 전체: {total_workers}명")
    print(f"   - 헬멧 착용: {helmet_count}명 ✅")
    print(f"   - 헬멧 미착용: {head_count}명 ⚠️")
    print(f"   - 안전조끼: {vest_count}개")

    if total_workers > 0:
        print(f"\n📊 안전 지표:")
        print(f"   - 헬멧 착용률: {helmet_rate:.1f}%")
        print(f"   - 안전 상태: {status}")

    if head_count > 0:
        print(f"\n🚨 경고:")
        print(f"   {head_count}명의 작업자가 헬멧을 착용하지 않았습니다!")
        print(f"   즉시 헬멧 착용을 지시하세요!")

    print(f"\n💾 저장 위치: {output_path}")

    return detection_info, helmet_rate if total_workers > 0 else 100

def main():
    print("\n" + "="*70)
    print("🔍 PPE Detection - Head(헬멧 미착용) 클래스 탐지 데모")
    print("="*70)

    # 경로 설정
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / 'models' / 'ppe_detection' / 'weights' / 'best.pt'
    test_dir = base_dir / 'dataset' / 'data' / 'test' / 'images'
    output_dir = base_dir / 'output' / 'head_detection'

    output_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    print("\n🤖 모델 로드 중...")
    model = YOLO(str(model_path))
    print("   ✅ 모델 로드 완료")

    # Head 클래스가 많이 포함될 가능성이 높은 이미지 선택
    test_images = [
        'ds1_hard_hat_workers0.png',      # 이미 확인된 head 포함
        'ds1_hard_hat_workers140.png',    # 추가 테스트
        'ds1_hard_hat_workers1302.png',   # 추가 테스트
        'ds1_hard_hat_workers2149.png',   # 추가 테스트
        'ds1_hard_hat_workers4307.png',   # 추가 테스트
    ]

    print(f"\n📸 {len(test_images)}개 이미지 분석 중...")

    total_results = []

    # 각 이미지 처리
    for img_name in test_images:
        img_path = test_dir / img_name
        if img_path.exists():
            detections, compliance = detect_and_warn(model, img_path, output_dir)
            total_results.append({
                'image': img_name,
                'detections': detections,
                'compliance': compliance
            })
        else:
            print(f"\n⚠️ 이미지를 찾을 수 없습니다: {img_name}")

    # 전체 통계
    print("\n" + "="*70)
    print("📊 전체 분석 결과")
    print("="*70)

    total_helmet = sum(
        sum(1 for d in r['detections'] if d['class'] == 'helmet')
        for r in total_results if r['detections']
    )
    total_head = sum(
        sum(1 for d in r['detections'] if d['class'] == 'head')
        for r in total_results if r['detections']
    )

    print(f"🏗️ 전체 작업자: {total_helmet + total_head}명")
    print(f"✅ 헬멧 착용: {total_helmet}명")
    print(f"⚠️ 헬멧 미착용: {total_head}명")

    if total_helmet + total_head > 0:
        overall_compliance = total_helmet / (total_helmet + total_head) * 100
        print(f"📈 전체 헬멧 착용률: {overall_compliance:.1f}%")

        if overall_compliance >= 90:
            print("🎯 전체 안전 수준: ✅ 우수")
        elif overall_compliance >= 70:
            print("🎯 전체 안전 수준: ⚠️ 주의 필요")
        else:
            print("🎯 전체 안전 수준: 🚨 즉시 조치 필요!")

    print(f"\n📂 결과 저장 위치: {output_dir}")
    print("="*70)

if __name__ == '__main__':
    main()