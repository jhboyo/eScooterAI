"""
훈련 결과 시각화 스크립트

## 이 스크립트는?
models/ppe_detection/results.csv 파일을 읽어서 훈련 결과를 시각화합니다.
손실 그래프, mAP 그래프, Precision/Recall 그래프를 생성합니다.

## 사용 방법
```bash
uv run python src/2_training/visualize_results.py
```

## 출력 파일
- models/ppe_detection/training_curves.png: 훈련 결과 그래프
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv


def visualize_training_results(results_path=None, save_path=None):
    """
    훈련 결과 시각화

    Args:
        results_path: results.csv 파일 경로
        save_path: 그래프 저장 경로
    """

    # 기본 경로 설정
    base_dir = Path(__file__).parent.parent.parent

    # .env 파일 로드
    env_path = base_dir / '.env'
    load_dotenv(env_path)

    project_root = os.getenv('PROJECT_ROOT', str(base_dir))

    if results_path is None:
        results_path = Path(project_root) / 'models' / 'ppe_detection' / 'results.csv'

    if save_path is None:
        save_path = Path(project_root) / 'models' / 'ppe_detection' / 'training_curves.png'

    # 파일 존재 확인
    if not Path(results_path).exists():
        print(f"결과 파일을 찾을 수 없습니다: {results_path}")
        return

    # 데이터 로드
    print(f"결과 파일 로드 중: {results_path}")
    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip()  # 컬럼명 공백 제거

    print(f"총 {len(df)} epochs 데이터 로드 완료")

    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('YOLOv8 PPE Detection Training Results', fontsize=14, fontweight='bold')

    # 1. 훈련 손실 그래프
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train/box_loss'], label='Box Loss', marker='o', markersize=3)
    ax1.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', marker='s', markersize=3)
    ax1.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', marker='^', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 검증 손실 그래프
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['val/box_loss'], label='Box Loss', marker='o', markersize=3)
    ax2.plot(df['epoch'], df['val/cls_loss'], label='Class Loss', marker='s', markersize=3)
    ax2.plot(df['epoch'], df['val/dfl_loss'], label='DFL Loss', marker='^', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. mAP 그래프
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['metrics/mAP50(B)'] * 100, label='mAP@0.5', marker='o', markersize=3, color='green')
    ax3.plot(df['epoch'], df['metrics/mAP50-95(B)'] * 100, label='mAP@0.5:0.95', marker='s', markersize=3, color='blue')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('mAP (%)')
    ax3.set_title('Mean Average Precision')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Precision/Recall 그래프
    ax4 = axes[1, 1]
    ax4.plot(df['epoch'], df['metrics/precision(B)'] * 100, label='Precision', marker='o', markersize=3, color='orange')
    ax4.plot(df['epoch'], df['metrics/recall(B)'] * 100, label='Recall', marker='s', markersize=3, color='red')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Score (%)')
    ax4.set_title('Precision & Recall')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 저장
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"그래프 저장 완료: {save_path}")

    # 화면에 표시
    plt.show()

    # 최종 결과 출력
    print("\n" + "=" * 50)
    print("최종 Epoch 결과")
    print("=" * 50)
    last = df.iloc[-1]
    print(f"Epoch: {int(last['epoch'])}")
    print(f"Train Loss: {last['train/box_loss']:.4f} (box) + {last['train/cls_loss']:.4f} (cls) + {last['train/dfl_loss']:.4f} (dfl)")
    print(f"Val Loss: {last['val/box_loss']:.4f} (box) + {last['val/cls_loss']:.4f} (cls) + {last['val/dfl_loss']:.4f} (dfl)")
    print(f"mAP@0.5: {last['metrics/mAP50(B)'] * 100:.2f}%")
    print(f"mAP@0.5:0.95: {last['metrics/mAP50-95(B)'] * 100:.2f}%")
    print(f"Precision: {last['metrics/precision(B)'] * 100:.2f}%")
    print(f"Recall: {last['metrics/recall(B)'] * 100:.2f}%")


if __name__ == '__main__':
    visualize_training_results()
