"""
Step 1: Pascal VOC (XML) → YOLO (TXT) 변환

Dataset 1 (kaggle-safey_helmet)의 XML 라벨을 YOLO 형식으로 변환합니다.

## Pascal VOC 형식이란?
- XML 기반의 객체 탐지 라벨 형식
- 바운딩 박스를 절대 좌표(픽셀)로 표현: xmin, ymin, xmax, ymax
- 각 이미지마다 하나의 XML 파일

## YOLO 형식이란?
- TXT 기반의 객체 탐지 라벨 형식
- 바운딩 박스를 정규화된 상대 좌표(0~1)로 표현: x_center, y_center, width, height
- 각 이미지마다 하나의 TXT 파일
- 한 줄에 하나의 객체: class_id x_center y_center width height

## 변환 공식
- x_center = (xmin + xmax) / 2 / image_width
- y_center = (ymin + ymax) / 2 / image_height
- width = (xmax - xmin) / image_width
- height = (ymax - ymin) / image_height

## 클래스 매핑 (3 Class)
- helmet → 0 (헬멧 착용)
- head → 1 (헬멧 미착용 - 추가!)
- person → -1 (제외: 전신 탐지 불필요)

## 입력/출력
- 입력: dataset/raw_data/raw/kaggle-safey_helmet/annotations/*.xml
- 출력: dataset/raw_data/processed/dataset1/labels/*.txt
"""

import xml.etree.ElementTree as ET
import os
from pathlib import Path
import shutil

# =============================================================================
# 클래스 매핑 정의 (3 Class)
# =============================================================================
# Dataset 1의 원본 클래스를 우리 프로젝트의 클래스 ID로 매핑
# -1은 해당 클래스를 제외한다는 의미
CLASS_MAPPING = {
    'helmet': 0,   # 헬멧 착용 → class 0
    'head': 1,     # 헬멧 미착용 (머리만) → class 1 (변경!)
    'person': -1   # 전신 → 제외
}


def convert_voc_to_yolo(xml_file: str, output_dir: str) -> bool:
    """
    단일 Pascal VOC XML 파일을 YOLO TXT 형식으로 변환

    변환 과정:
    1. XML 파일 파싱
    2. 이미지 크기 정보 추출 (정규화에 필요)
    3. 각 객체(object)에 대해:
       - 클래스명 확인 및 매핑
       - 바운딩 박스 좌표 추출 (xmin, ymin, xmax, ymax)
       - YOLO 형식으로 변환 (정규화된 중심점 + 크기)
    4. 유효한 객체가 있으면 TXT 파일로 저장

    Args:
        xml_file: 변환할 XML 파일의 전체 경로
        output_dir: 변환된 TXT 파일을 저장할 디렉토리

    Returns:
        bool: 변환 성공 여부
              - True: 유효한 객체(helmet)가 있어서 파일 생성됨
              - False: 유효한 객체가 없거나 에러 발생
    """
    try:
        # -----------------------------------------------------------------
        # 1. XML 파일 파싱
        # -----------------------------------------------------------------
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # -----------------------------------------------------------------
        # 2. 이미지 크기 정보 추출
        # -----------------------------------------------------------------
        # YOLO 형식은 0~1 사이의 정규화된 좌표를 사용하므로
        # 원본 이미지 크기가 필요함
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        # 출력 파일 경로 설정 (확장자만 .xml → .txt로 변경)
        filename = Path(xml_file).stem + '.txt'
        output_path = os.path.join(output_dir, filename)

        # 변환된 라벨을 저장할 리스트
        lines = []

        # -----------------------------------------------------------------
        # 3. 모든 객체(object) 처리
        # -----------------------------------------------------------------
        for obj in root.findall('object'):
            # 클래스명 추출
            class_name = obj.find('name').text

            # 클래스 매핑 확인 (정의되지 않은 클래스는 무시)
            if class_name not in CLASS_MAPPING:
                continue

            # 클래스 ID 가져오기
            class_id = CLASS_MAPPING[class_name]

            # 제외 클래스(-1)는 스킵
            if class_id == -1:
                continue

            # ---------------------------------------------------------
            # 바운딩 박스 좌표 추출 (픽셀 단위)
            # ---------------------------------------------------------
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)  # 왼쪽 상단 x
            ymin = int(bbox.find('ymin').text)  # 왼쪽 상단 y
            xmax = int(bbox.find('xmax').text)  # 오른쪽 하단 x
            ymax = int(bbox.find('ymax').text)  # 오른쪽 하단 y

            # ---------------------------------------------------------
            # YOLO 형식으로 변환 (정규화된 중심점 + 크기)
            # ---------------------------------------------------------
            # 중심점 x = (xmin + xmax) / 2 / 이미지너비
            x_center = (xmin + xmax) / 2.0 / img_width
            # 중심점 y = (ymin + ymax) / 2 / 이미지높이
            y_center = (ymin + ymax) / 2.0 / img_height
            # 너비 = (xmax - xmin) / 이미지너비
            width = (xmax - xmin) / img_width
            # 높이 = (ymax - ymin) / 이미지높이
            height = (ymax - ymin) / img_height

            # YOLO 형식 문자열 생성
            # 형식: class_id x_center y_center width height
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # -----------------------------------------------------------------
        # 4. 파일 저장
        # -----------------------------------------------------------------
        # 유효한 객체(helmet 또는 head)가 있는 경우에만 파일 저장
        if lines:
            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))
            return True

        # 유효한 객체가 없으면 False 반환 (파일 생성 안 함)
        return False

    except Exception as e:
        print(f"❌ 변환 실패: {xml_file} - {str(e)}")
        return False


def convert_dataset1():
    """
    Dataset 1 전체 변환 실행

    처리 과정:
    1. 입출력 경로 설정
    2. 출력 디렉토리 생성
    3. 모든 XML 파일에 대해:
       - VOC → YOLO 변환
       - 변환 성공 시 해당 이미지도 복사
    4. 결과 통계 출력
    """

    # =========================================================================
    # 1. 경로 설정
    # =========================================================================
    # 프로젝트 루트 디렉토리 (step1_convert_voc_to_yolo.py 기준으로 3단계 상위)
    base_dir = Path(__file__).parent.parent.parent

    # 입력 경로: 원본 데이터셋 위치
    input_dir = base_dir / 'dataset' / 'raw_data' / 'raw' / 'kaggle-safey_helmet'
    annotations_dir = input_dir / 'annotations'  # XML 파일들
    images_dir = input_dir / 'images'            # 이미지 파일들

    # 출력 경로: 변환된 데이터 저장 위치
    output_dir = base_dir / 'dataset' / 'raw_data' / 'processed' / 'dataset1'
    output_images_dir = output_dir / 'images'    # 이미지 저장
    output_labels_dir = output_dir / 'labels'    # 라벨(TXT) 저장

    # =========================================================================
    # 2. 출력 디렉토리 생성
    # =========================================================================
    # parents=True: 중간 디렉토리도 함께 생성
    # exist_ok=True: 이미 존재해도 에러 발생 안 함
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 3. 변환 시작
    # =========================================================================
    print("=" * 50)
    print("Step 1: Dataset 1 VOC → YOLO 변환")
    print("=" * 50)
    print(f"입력 경로: {input_dir}")
    print(f"출력 경로: {output_dir}")
    print()

    # 모든 XML 파일 목록 가져오기
    xml_files = list(annotations_dir.glob('*.xml'))
    total = len(xml_files)

    print(f"총 {total}개 파일 변환 중...")

    # 통계 변수
    converted = 0  # 변환 성공 (helmet 또는 head가 있는 이미지)
    skipped = 0    # 스킵됨 (유효한 객체가 없는 이미지)

    # =========================================================================
    # 4. 각 XML 파일 처리
    # =========================================================================
    for i, xml_file in enumerate(xml_files, 1):
        # VOC → YOLO 변환 시도
        if convert_voc_to_yolo(str(xml_file), str(output_labels_dir)):
            # 변환 성공: 해당 이미지도 복사
            # 이미지 파일명 = XML 파일명 + .png
            image_name = xml_file.stem + '.png'
            src_image = images_dir / image_name
            dst_image = output_images_dir / image_name

            if src_image.exists():
                # 이미지 파일 복사
                shutil.copy(src_image, dst_image)
                converted += 1
            else:
                # PNG가 없으면 JPG/JPEG 확장자 시도
                for ext in ['.jpg', '.jpeg']:
                    alt_image = images_dir / (xml_file.stem + ext)
                    if alt_image.exists():
                        shutil.copy(alt_image, output_images_dir / alt_image.name)
                        converted += 1
                        break
        else:
            # 변환 실패 (유효한 객체가 없음)
            skipped += 1

        # 진행 상황 출력 (500개마다 또는 마지막)
        if i % 500 == 0 or i == total:
            print(f"  진행: {i}/{total} ({i*100//total}%)")

    # =========================================================================
    # 5. 결과 출력
    # =========================================================================
    print()
    print(f"✅ 변환 완료!")
    print(f"   - 변환됨: {converted}개")
    print(f"   - 스킵됨 (유효 객체 없음): {skipped}개")
    print(f"   - 출력 위치: {output_dir}")
    print()

    return converted, skipped


if __name__ == '__main__':
    # 스크립트 직접 실행 시 변환 수행
    convert_dataset1()
