"""
마크다운 파일을 PDF로 변환하는 스크립트
"""
import sys
import markdown
from weasyprint import HTML, CSS
from pathlib import Path


def convert_md_to_pdf(md_file: str, output_pdf: str = None):
    """
    마크다운 파일을 PDF로 변환

    Args:
        md_file: 입력 마크다운 파일 경로
        output_pdf: 출력 PDF 파일 경로 (None이면 같은 이름으로 생성)
    """
    md_path = Path(md_file)

    if not md_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {md_file}")
        return False

    # 출력 파일명 결정
    if output_pdf is None:
        output_pdf = md_path.with_suffix('.pdf')
    else:
        output_pdf = Path(output_pdf)

    print(f"📄 변환 중: {md_path.name} → {output_pdf.name}")

    try:
        # 마크다운 파일 읽기
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # 마크다운을 HTML로 변환
        md = markdown.Markdown(
            extensions=[
                'extra',        # 테이블, 코드 블록 등 확장 문법
                'codehilite',   # 코드 하이라이팅
                'toc',          # 목차
                'tables',       # 테이블
                'fenced_code',  # 펜스 코드 블록
            ]
        )
        html_content = md.convert(md_content)

        # HTML 템플릿 생성
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{md_path.stem}</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm 1.5cm;
        }}
        body {{
            font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 100%;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            font-size: 2em;
            margin-top: 1em;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            font-size: 1.5em;
            margin-top: 1.5em;
        }}
        h3 {{
            color: #555;
            font-size: 1.3em;
            margin-top: 1.2em;
        }}
        h4 {{
            color: #666;
            font-size: 1.1em;
            margin-top: 1em;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
            font-size: 0.9em;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 10px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 15px;
            color: #555;
            font-style: italic;
            margin: 1em 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1em auto;
            page-break-inside: avoid;
        }}
        ul, ol {{
            margin: 0.5em 0;
            padding-left: 2em;
        }}
        li {{
            margin: 0.3em 0;
        }}
        strong {{
            color: #2c3e50;
        }}
        em {{
            color: #555;
        }}
        hr {{
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 2em 0;
        }}
        .page-break {{
            page-break-after: always;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

        # HTML을 PDF로 변환
        HTML(string=html_template).write_pdf(
            output_pdf,
            stylesheets=[CSS(string='@page { size: A4; margin: 2cm 1.5cm; }')]
        )

        print(f"✅ 변환 완료: {output_pdf}")
        print(f"   파일 크기: {output_pdf.stat().st_size / 1024:.1f} KB")
        return True

    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("사용법: python convert_md_to_pdf.py <markdown_file> [output_pdf]")
        print("예제: python convert_md_to_pdf.py 모델학습결과보고서.md")
        sys.exit(1)

    md_file = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else None

    success = convert_md_to_pdf(md_file, output_pdf)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
