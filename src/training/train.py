"""
YOLOv8 모델 훈련 스크립트 (3 Class)

## 이 스크립트는?
YOLOv8 모델을 PPE 데이터셋(3 class)으로 훈련하는 메인 훈련 스크립트입니다.
Transfer Learning을 활용하여 COCO 사전학습 가중치를 사용합니다.

## 탐지 클래스
- Class 0: helmet (헬멧 착용) ✅
- Class 1: head (헬멧 미착용) ⚠️
- Class 2: vest (안전조끼 착용) ✅

## 사용 방법
```bash
# 기본 실행 (configs/train_config.yaml 사용)
uv run python src/training/train.py

# MacBook 테스트 훈련 (3 epochs, batch 16)
uv run python src/training/train.py --epochs 1 --batch 16 --device mps

# RunPod A100 본 훈련 (100 epochs, batch 128)
uv run python src/training/train.py --epochs 100 --batch 128 --device 0

# 다른 설정 파일 사용
uv run python src/training/train.py --config configs/custom_config.yaml

# 데이터셋 직접 지정
uv run python src/training/train.py --data configs/ppe_dataset.yaml
```

## 실행 과정
1. YOLOv8 모델 로드 (yolov8n.pt - Nano 버전)
2. configs/train_config.yaml에서 하이퍼파라미터 로드
3. configs/ppe_dataset.yaml에서 데이터셋 정보 로드
4. 훈련 실행 및 모델 저장
5. 훈련 결과 및 통계 출력

## 출력 파일
훈련 완료 후 models/ppe_detection/ 폴더에 다음 파일들이 생성됩니다:
- weights/best.pt: 최고 성능 모델
- weights/last.pt: 마지막 체크포인트
- results.csv: 에포크별 훈련 통계
- confusion_matrix.png: 혼동 행렬
- PR_curve.png: Precision-Recall 곡선
- results.png: 훈련 결과 그래프
"""

import argparse
import os
from pathlib import Path
import yaml
import torch
from dotenv import load_dotenv
from ultralytics import YOLO


def check_device():
    """
    사용 가능한 디바이스 확인

    Returns:
        str: 사용할 디바이스 ('cuda', 'mps', 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA GPU 사용 가능: {device_name}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Apple Silicon MPS 사용 가능")
    else:
        device = 'cpu'
        print("GPU 사용 불가, CPU로 훈련합니다.")

    return device


def load_config(config_path):
    """
    훈련 설정 파일 로드

    Args:
        config_path: train_config.yaml 파일 경로

    Returns:
        dict: 설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train(args):
    """
    YOLOv8 모델 훈련 실행

    Args:
        args: 명령줄 인자 (argparse Namespace)

    실행 순서:
    1. 디바이스 확인
    2. 설정 파일 로드
    3. YOLOv8 모델 로드
    4. 훈련 실행
    5. 결과 출력
    """

    # =========================================================================
    # 1. 초기화
    # =========================================================================
    print("=" * 60)
    print("YOLOv8 PPE Detection 모델 훈련 (3 Class)")
    print("=" * 60)
    print()

    # =========================================================================
    # 2. 설정 파일 로드
    # =========================================================================
    print("설정 파일 로드 중...")
    config = load_config(args.config)

    # 설정값 추출 (명령줄 인자 우선)
    model_name = config['model']['name']
    data_yaml = args.data if args.data else config['data']['yaml']
    epochs = args.epochs if args.epochs is not None else config['train']['epochs']
    batch_size = args.batch if args.batch is not None else config['train']['batch_size']
    img_size = config['train']['img_size']
    patience = config['train']['patience']
    workers = config['train'].get('workers', 8)  # 데이터 로드 스레드 수

    # 디바이스 설정 (명령줄 인자 우선)
    if args.device:
        device = args.device
    elif config.get('device'):
        device = config['device']
    else:
        device = check_device()

    # Optimizer 설정
    optimizer = config['train']['optimizer']
    lr0 = config['train']['lr0']
    lrf = config['train']['lrf']
    momentum = config['train']['momentum']
    weight_decay = config['train']['weight_decay']
    warmup_epochs = config['train']['warmup_epochs']

    # 출력 설정
    save_dir = config['output']['save_dir']
    name = config['output']['name']
    exist_ok = config['output']['exist_ok']

    # 증강 설정
    augment = config['augment']

    print(f"   - 모델: {model_name}")
    print(f"   - 데이터셋: {data_yaml}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Image Size: {img_size}")
    print(f"   - Device: {device}")
    print()

    # =========================================================================
    # 3. YOLOv8 모델 로드
    # =========================================================================
    print("YOLOv8 모델 로드 중...")

    # 사전학습 가중치 로드 (Transfer Learning)
    # yolov8n.pt: COCO 데이터셋으로 학습된 Nano 버전
    model = YOLO(f"{model_name}.pt")

    print(f"   {model_name}.pt 로드 완료")
    print()

    # =========================================================================
    # 4. 훈련 실행
    # =========================================================================
    print("훈련 시작...")
    print("-" * 60)

    # 훈련 실행
    # 모든 하이퍼파라미터를 train_config.yaml에서 가져옴
    results = model.train(
        # 데이터셋 설정
        data=data_yaml,

        # 기본 훈련 설정
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=patience,
        workers=workers,

        # Optimizer 설정
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_momentum=config['train']['warmup_momentum'],
        warmup_bias_lr=config['train']['warmup_bias_lr'],

        # 증강 설정
        hsv_h=augment['hsv_h'],
        hsv_s=augment['hsv_s'],
        hsv_v=augment['hsv_v'],
        degrees=augment['degrees'],
        translate=augment['translate'],
        scale=augment['scale'],
        shear=augment['shear'],
        flipud=augment['flipud'],
        fliplr=augment['fliplr'],
        mosaic=augment['mosaic'],
        mixup=augment['mixup'],

        # 출력 설정
        project=save_dir,
        name=name,
        exist_ok=exist_ok,

        # 디바이스 설정
        device=device,

        # 로그 설정
        verbose=config['verbose'],
        save_period=config['save_period'],

        # 기타 설정
        save=True,          # 체크포인트 저장
        plots=True,         # 훈련 그래프 저장
        val=True,           # 훈련 중 검증 수행
    )

    # =========================================================================
    # 5. 결과 출력
    # =========================================================================
    print()
    print("=" * 60)
    print("훈련 완료!")
    print("=" * 60)

    # 결과 파일 경로
    output_path = Path(save_dir) / name
    weights_path = output_path / 'weights'

    print(f"\n결과 파일 위치: {output_path}")
    print(f"   - 최고 성능 모델: {weights_path / 'best.pt'}")
    print(f"   - 마지막 체크포인트: {weights_path / 'last.pt'}")
    print(f"   - 훈련 통계: {output_path / 'results.csv'}")
    print(f"   - 혼동 행렬: {output_path / 'confusion_matrix.png'}")
    print(f"   - PR 곡선: {output_path / 'PR_curve.png'}")
    print()

    # 최고 성능 출력
    if hasattr(results, 'results_dict'):
        print("최고 성능 통계:")
        metrics = results.results_dict
        if 'metrics/mAP50(B)' in metrics:
            print(f"   - mAP@0.5: {metrics['metrics/mAP50(B)']:.4f}")
        if 'metrics/mAP50-95(B)' in metrics:
            print(f"   - mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.4f}")
        if 'metrics/precision(B)' in metrics:
            print(f"   - Precision: {metrics['metrics/precision(B)']:.4f}")
        if 'metrics/recall(B)' in metrics:
            print(f"   - Recall: {metrics['metrics/recall(B)']:.4f}")

    print()
    print("다음 단계:")
    print("   1. 모델 평가: uv run python src/4_test/evaluate.py")
    print("   2. 추론 테스트: uv run python src/3_inference/inference.py --source test_image.jpg")
    print()

    return results


def main():
    """
    메인 함수 - 명령줄 인자 파싱 및 훈련 실행
    """
    # 기본 경로 설정
    # __file__ -> src/training/train.py
    # parent -> src/training/
    # parent.parent -> src/
    # parent.parent.parent -> 프로젝트 루트
    base_dir = Path(__file__).parent.parent.parent

    # .env 파일 로드
    env_path = base_dir / '.env'
    load_dotenv(env_path)

    # 환경 변수에서 프로젝트 루트 경로 가져오기
    project_root = os.getenv('PROJECT_ROOT', str(base_dir))

    default_config = Path(project_root) / 'configs' / 'train_config.yaml'

    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(
        description='YOLOv8 PPE Detection 모델 훈련',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default=str(default_config),
        help='훈련 설정 파일 경로 (default: configs/train_config.yaml)'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='데이터셋 YAML 파일 경로 (default: configs/ppe_dataset.yaml)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='훈련 에포크 수 (default: train_config.yaml 값 사용)'
    )

    parser.add_argument(
        '--batch',
        type=int,
        default=None,
        help='배치 크기 (default: train_config.yaml 값 사용)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='디바이스 설정 (예: cpu, 0, mps) (default: 자동 감지)'
    )

    args = parser.parse_args()

    # 훈련 실행
    train(args)


if __name__ == '__main__':
    main()
