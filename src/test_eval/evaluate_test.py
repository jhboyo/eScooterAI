"""
YOLOv8 모델 Test Dataset 평가 스크립트 (3 Class)

## 이 스크립트는?
훈련된 YOLOv8 모델을 Test Dataset으로 최종 평가하는 스크립트입니다.
Validation set이 아닌 Test set으로 모델의 일반화 성능을 검증합니다.

## 탐지 클래스
- Class 0: helmet (헬멧 착용) ✅
- Class 1: head (헬멧 미착용) ⚠️
- Class 2: vest (안전조끼 착용) ✅

## 사용 방법
```bash
# 기본 실행 (best.pt 모델, test dataset)
uv run python src/4_test/evaluate_test.py

# 특정 모델 지정
uv run python src/4_test/evaluate_test.py --model models/ppe_detection/weights/last.pt

# Confidence threshold 변경
uv run python src/4_test/evaluate_test.py --conf 0.25

# IoU threshold 변경
uv run python src/4_test/evaluate_test.py --iou 0.6
```

## 평가 항목
1. mAP@0.5, mAP@0.5:0.95
2. Precision, Recall
3. Confusion Matrix
4. 클래스별 AP (Average Precision)
5. Validation vs Test 성능 비교

## 출력 파일
평가 완료 후 다음 파일들이 생성됩니다:
- test_results/confusion_matrix.png: Test set 혼동 행렬
- test_results/BoxPR_curve.png: Precision-Recall 곡선
- test_results/results.csv: 상세 평가 결과
- test_report.md: 최종 평가 보고서
"""

import argparse
import os
from pathlib import Path
import yaml
from ultralytics import YOLO
import pandas as pd
from datetime import datetime


def load_validation_results(model_dir):
    """
    Validation set 결과 로드 (비교용)

    Args:
        model_dir: 모델 디렉토리 경로

    Returns:
        dict: Validation 결과 딕셔너리
    """
    results_csv = Path(model_dir) / 'results.csv'

    if not results_csv.exists():
        print("⚠️ Validation 결과 파일을 찾을 수 없습니다.")
        return None

    # 마지막 epoch의 결과 읽기
    df = pd.read_csv(results_csv)
    last_row = df.iloc[-1]

    return {
        'mAP50': last_row['metrics/mAP50(B)'],
        'mAP50-95': last_row['metrics/mAP50-95(B)'],
        'precision': last_row['metrics/precision(B)'],
        'recall': last_row['metrics/recall(B)']
    }


def evaluate_test_set(args):
    """
    Test Dataset으로 모델 평가 실행

    Args:
        args: 명령줄 인자 (argparse Namespace)

    실행 순서:
    1. 모델 로드
    2. Test set 평가
    3. 결과 분석 및 저장
    4. Validation vs Test 비교
    """

    # =========================================================================
    # 1. 초기화
    # =========================================================================
    print("=" * 70)
    print("YOLOv8 모델 Test Dataset 평가 (3 Class)")
    print("=" * 70)
    print()

    # 모델 경로 확인
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return

    # 데이터셋 설정 파일 확인
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"❌ 데이터셋 설정 파일을 찾을 수 없습니다: {data_yaml}")
        return

    print(f"📦 모델: {model_path}")
    print(f"📄 데이터셋: {data_yaml}")
    print(f"🎯 Confidence threshold: {args.conf}")
    print(f"📐 IoU threshold: {args.iou}")
    print()

    # =========================================================================
    # 2. 모델 로드
    # =========================================================================
    print("🤖 YOLOv8 모델 로드 중...")
    model = YOLO(str(model_path))
    print("   ✅ 모델 로드 완료")
    print()

    # =========================================================================
    # 3. Test Set 평가
    # =========================================================================
    print("🧪 Test Dataset 평가 시작...")
    print("-" * 70)

    # Test set으로 평가 (split='test' 지정)
    results = model.val(
        data=str(data_yaml),
        split='test',  # test dataset 사용
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        save_json=True,  # COCO JSON 형식으로 저장
        save_hybrid=True,  # 하이브리드 라벨 저장
        plots=True,  # 그래프 생성
        verbose=True
    )

    print()
    print("=" * 70)
    print("✅ Test Dataset 평가 완료!")
    print("=" * 70)
    print()

    # =========================================================================
    # 4. 결과 출력
    # =========================================================================
    print("📊 Test Set 성능 지표:")
    print("-" * 70)

    # 메트릭 추출
    test_metrics = {
        'mAP50': results.box.map50,
        'mAP50-95': results.box.map,
        'precision': results.box.mp,
        'recall': results.box.mr
    }

    print(f"   📈 mAP@0.5: {test_metrics['mAP50']:.4f} ({test_metrics['mAP50']*100:.2f}%)")
    print(f"   📈 mAP@0.5:0.95: {test_metrics['mAP50-95']:.4f} ({test_metrics['mAP50-95']*100:.2f}%)")
    print(f"   🎯 Precision: {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"   🔍 Recall: {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%)")
    print()

    # 클래스별 AP
    if hasattr(results.box, 'ap_class_index'):
        print("📊 클래스별 성능:")
        print("-" * 70)
        class_names = ['helmet', 'head', 'vest']

        # ap50 추출
        ap50_per_class = results.box.ap50

        for idx, class_name in enumerate(class_names):
            if idx < len(ap50_per_class):
                ap_value = ap50_per_class[idx]
                print(f"   {class_name:8s}: AP@0.5 = {ap_value:.4f} ({ap_value*100:.2f}%)")
        print()

    # =========================================================================
    # 5. Validation vs Test 비교
    # =========================================================================
    model_dir = model_path.parent.parent
    val_metrics = load_validation_results(model_dir)

    if val_metrics:
        print("📊 Validation vs Test 성능 비교:")
        print("-" * 70)
        print(f"{'지표':<20} {'Validation':>15} {'Test':>15} {'차이':>15}")
        print("-" * 70)

        for metric_name in ['mAP50', 'mAP50-95', 'precision', 'recall']:
            val_value = val_metrics[metric_name]
            test_value = test_metrics[metric_name]
            diff = test_value - val_value
            diff_pct = (diff / val_value * 100) if val_value != 0 else 0

            print(f"{metric_name:<20} {val_value:>14.4f} {test_value:>14.4f} {diff:>+14.4f} ({diff_pct:+.2f}%)")

        print("-" * 70)
        print()

        # 과적합 판단
        mAP_diff = test_metrics['mAP50'] - val_metrics['mAP50']
        if abs(mAP_diff) < 0.02:  # 2% 이내 차이
            print("✅ 판정: 일반화 성능 우수 (Validation ≈ Test)")
        elif mAP_diff < -0.05:  # 5% 이상 하락
            print("⚠️ 판정: 과적합 가능성 있음 (Test < Validation)")
        else:
            print("✅ 판정: 정상 범위")
        print()

    # =========================================================================
    # 6. 결과 파일 저장
    # =========================================================================
    print("💾 결과 파일 저장 중...")

    # 결과 디렉토리 생성
    output_dir = Path('output/test_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 결과 CSV 저장
    results_data = {
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'model': [str(model_path)],
        'test_mAP50': [test_metrics['mAP50']],
        'test_mAP50-95': [test_metrics['mAP50-95']],
        'test_precision': [test_metrics['precision']],
        'test_recall': [test_metrics['recall']]
    }

    if val_metrics:
        results_data['val_mAP50'] = [val_metrics['mAP50']]
        results_data['val_mAP50-95'] = [val_metrics['mAP50-95']]
        results_data['val_precision'] = [val_metrics['precision']]
        results_data['val_recall'] = [val_metrics['recall']]

    df = pd.DataFrame(results_data)
    csv_path = output_dir / 'test_evaluation_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"   ✅ CSV 저장: {csv_path}")

    # Validation 결과와 함께 시각화 파일 이동
    # YOLOv8이 자동 생성한 파일들을 test_results로 이동
    runs_dir = Path('runs/detect')
    if runs_dir.exists():
        latest_dir = max(runs_dir.glob('val*'), key=os.path.getctime, default=None)
        if latest_dir:
            import shutil
            for img_file in ['confusion_matrix.png', 'confusion_matrix_normalized.png',
                           'BoxPR_curve.png', 'BoxF1_curve.png',
                           'BoxP_curve.png', 'BoxR_curve.png']:
                src = latest_dir / img_file
                if src.exists():
                    dst = output_dir / f'test_{img_file}'
                    shutil.copy(src, dst)
                    print(f"   ✅ 시각화 저장: {dst}")

    print()
    print("=" * 70)
    print("🎉 모든 평가 완료!")
    print("=" * 70)
    print()
    print("📂 결과 파일 위치:")
    print(f"   📁 {output_dir.absolute()}")
    print()
    print("다음 단계:")
    print("   1️⃣ test_results/ 폴더에서 결과 확인")
    print("   2️⃣ Confusion Matrix 분석")
    print("   3️⃣ 최종 보고서 작성")
    print()

    return results


def main():
    """
    메인 함수 - 명령줄 인자 파싱 및 평가 실행
    """
    # 기본 경로 설정
    base_dir = Path(__file__).parent.parent.parent

    default_model = base_dir / 'models' / 'ppe_detection' / 'weights' / 'best.pt'
    default_data = base_dir / 'configs' / 'ppe_dataset.yaml'

    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(
        description='YOLOv8 PPE Detection 모델 Test Dataset 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        default=str(default_model),
        help='평가할 모델 파일 경로 (default: models/ppe_detection/weights/best.pt)'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=str(default_data),
        help='데이터셋 YAML 파일 경로 (default: configs/ppe_dataset.yaml)'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.001,
        help='Confidence threshold (default: 0.001 for evaluation)'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=0.6,
        help='IoU threshold for NMS (default: 0.6)'
    )

    parser.add_argument(
        '--batch',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )

    args = parser.parse_args()

    # 평가 실행
    evaluate_test_set(args)


if __name__ == '__main__':
    main()
