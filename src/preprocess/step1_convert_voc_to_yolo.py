"""
Step 1: Pascal VOC (XML) → YOLO (TXT) 변환

Dataset 1 (kaggle-safey_helmet)의 XML 라벨을 YOLO 형식으로 변환합니다.
- helmet 클래스만 추출 (class_id: 0)
- head, person 클래스는 제외
- 해당 Dataset에는 안전조끼는 없음
"""

import xml.etree.ElementTree as ET
import os
from pathlib import Path
import shutil

# 클래스 매핑
CLASS_MAPPING = {
    'helmet': 0,
    'head': -1,    # 제외
    'person': -1   # 제외
}

def convert_voc_to_yolo(xml_file: str, output_dir: str) -> bool:
    """
    Pascal VOC XML을 YOLO TXT로 변환

    Args:
        xml_file: XML 파일 경로
        output_dir: 출력 디렉토리

    Returns:
        bool: 변환 성공 여부 (유효한 객체가 있으면 True)
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 이미지 크기 가져오기
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        # 출력 파일명
        filename = Path(xml_file).stem + '.txt'
        output_path = os.path.join(output_dir, filename)

        lines = []

        # 모든 객체 처리
        for obj in root.findall('object'):
            class_name = obj.find('name').text

            # 클래스 매핑 확인
            if class_name not in CLASS_MAPPING:
                continue

            class_id = CLASS_MAPPING[class_name]

            # 제외 클래스는 스킵
            if class_id == -1:
                continue

            # 바운딩 박스 좌표
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # YOLO 형식으로 변환 (정규화된 중심점과 크기)
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 유효한 객체가 있으면 파일 저장
        if lines:
            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))
            return True

        return False

    except Exception as e:
        print(f"❌ 변환 실패: {xml_file} - {str(e)}")
        return False


def convert_dataset1():
    """Dataset 1 전체 변환"""

    # 경로 설정
    base_dir = Path(__file__).parent.parent.parent
    input_dir = base_dir / 'images' / 'raw' / 'kaggle-safey_helmet'
    output_dir = base_dir / 'images' / 'processed' / 'dataset1'

    annotations_dir = input_dir / 'annotations'
    images_dir = input_dir / 'images'

    output_images_dir = output_dir / 'images'
    output_labels_dir = output_dir / 'labels'

    # 출력 디렉토리 생성
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Step 1: Dataset 1 VOC → YOLO 변환")
    print("=" * 50)
    print(f"입력 경로: {input_dir}")
    print(f"출력 경로: {output_dir}")
    print()

    # XML 파일 목록
    xml_files = list(annotations_dir.glob('*.xml'))
    total = len(xml_files)

    print(f"총 {total}개 파일 변환 중...")

    converted = 0
    skipped = 0

    for i, xml_file in enumerate(xml_files, 1):
        # 변환
        if convert_voc_to_yolo(str(xml_file), str(output_labels_dir)):
            # 이미지 복사
            image_name = xml_file.stem + '.png'
            src_image = images_dir / image_name
            dst_image = output_images_dir / image_name

            if src_image.exists():
                shutil.copy(src_image, dst_image)
                converted += 1
            else:
                # PNG가 없으면 다른 확장자 시도
                for ext in ['.jpg', '.jpeg']:
                    alt_image = images_dir / (xml_file.stem + ext)
                    if alt_image.exists():
                        shutil.copy(alt_image, output_images_dir / alt_image.name)
                        converted += 1
                        break
        else:
            skipped += 1

        # 진행 상황 출력
        if i % 500 == 0 or i == total:
            print(f"  진행: {i}/{total} ({i*100//total}%)")

    print()
    print(f"✅ 변환 완료!")
    print(f"   - 변환됨: {converted}개")
    print(f"   - 스킵됨 (helmet 없음): {skipped}개")
    print(f"   - 출력 위치: {output_dir}")
    print()

    return converted, skipped


if __name__ == '__main__':
    convert_dataset1()
