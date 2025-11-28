"""
Vector Database Builder

전동킥보드 안전 문서를 FAISS 벡터 DB로 구축
법규, 가이드, 사례 문서를 임베딩하여 벡터 저장소 생성
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from .vector_store import FAISSVectorStore


def load_documents_from_json(json_path: str) -> List[Dict[str, any]]:
    """
    JSON 파일에서 문서 로드

    Args:
        json_path: JSON 파일 경로

    Returns:
        List[Dict]: 문서 리스트 (text, metadata)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"✅ Loaded {len(documents)} documents from {json_path}")
    return documents


def build_vector_database(
    docs_dir: str,
    output_dir: str,
    embedding_dimension: int = 1536,
    api_key: str = None
) -> FAISSVectorStore:
    """
    벡터 데이터베이스 구축

    1. 법규, 가이드, 사례 JSON 파일 로드
    2. 각 문서를 OpenAI 임베딩으로 벡터화
    3. FAISS 인덱스에 추가
    4. 디스크에 저장

    Args:
        docs_dir: 문서 디렉토리 경로
        output_dir: 벡터 DB 저장 경로
        embedding_dimension: 임베딩 차원 (기본: 1536)
        api_key: OpenAI API 키

    Returns:
        FAISSVectorStore: 구축된 벡터 저장소
    """
    print("\n" + "="*80)
    print("📚 전동킥보드 안전 교육 벡터 데이터베이스 구축")
    print("="*80 + "\n")

    # ========================================================================
    # 1. 벡터 저장소 초기화
    # ========================================================================
    # FAISS IndexFlatL2 기반 벡터 저장소 생성
    # - embedding_dimension: OpenAI text-embedding-3-small의 차원 (1536)
    # - L2 거리 기반 유사도 검색 준비
    print(f"🔧 벡터 저장소 초기화 (차원: {embedding_dimension})")
    vector_store = FAISSVectorStore(dimension=embedding_dimension, api_key=api_key)

    # ========================================================================
    # 2. 문서 로드 (Knowledge Base 구축)
    # ========================================================================
    # 안전 교육 지식 베이스를 3개 카테고리로 분류하여 로드
    # - laws.json: 법규 문서 (도로교통법, 과태료 등)
    # - guides.json: 안전 가이드 (헬멧 착용법, 운전 수칙 등)
    # - cases.json: 사고 사례 및 통계
    docs_path = Path(docs_dir)
    all_documents = []  # 전체 문서를 하나의 리스트로 통합

    # 법규 문서 로드
    laws_path = docs_path / "laws.json"
    if laws_path.exists():
        laws_docs = load_documents_from_json(str(laws_path))
        all_documents.extend(laws_docs)  # 전체 문서 리스트에 추가
    else:
        print(f"⚠️  Laws file not found: {laws_path}")

    # 가이드 문서 로드
    guides_path = docs_path / "guides.json"
    if guides_path.exists():
        guides_docs = load_documents_from_json(str(guides_path))
        all_documents.extend(guides_docs)
    else:
        print(f"⚠️  Guides file not found: {guides_path}")

    # 사례 문서 로드
    cases_path = docs_path / "cases.json"
    if cases_path.exists():
        cases_docs = load_documents_from_json(str(cases_path))
        all_documents.extend(cases_docs)
    else:
        print(f"⚠️  Cases file not found: {cases_path}")

    if not all_documents:
        raise ValueError("No documents found. Please check the documents directory.")

    print(f"\n📊 Total documents: {len(all_documents)}")

    # 카테고리별 통계
    category_stats = {}
    for doc in all_documents:
        category = doc["metadata"].get("category", "Unknown")
        category_stats[category] = category_stats.get(category, 0) + 1

    print("\n📈 Category Statistics:")
    for category, count in category_stats.items():
        print(f"  - {category}: {count} documents")

    # ========================================================================
    # 3. 임베딩 및 FAISS 인덱스 구축
    # ========================================================================
    # 각 문서를 OpenAI API로 임베딩하고 FAISS 인덱스에 추가
    # - 텍스트 → 1536차원 벡터 변환 (의미적 표현)
    # - FAISS IndexFlatL2에 벡터 저장
    # - 시간 복잡도: O(N * D) where N=문서 수, D=차원 수
    # - API 호출: N번 (문서당 1번)
    print(f"\n🔄 Embedding documents with OpenAI text-embedding-3-small...")
    print("   (This may take a few minutes depending on the number of documents)")

    vector_store.add_documents(all_documents)

    print(f"✅ Successfully embedded {len(all_documents)} documents")

    # ========================================================================
    # 4. 벡터 저장소 디스크에 저장
    # ========================================================================
    # FAISS 인덱스와 문서 데이터를 영구 저장
    # - {output_dir}/index.faiss: FAISS 벡터 인덱스 (바이너리)
    # - {output_dir}/documents.json: 원본 텍스트 + 메타데이터 (JSON)
    # 런타임에 load() 메서드로 불러와서 사용 가능
    print(f"\n💾 Saving vector database to {output_dir}")
    vector_store.save(output_dir)

    print("\n✅ Vector database built successfully!")
    print(f"\n📊 Final Statistics:")
    stats = vector_store.get_stats()
    for key, value in stats.items():
        print(f"  - {key}: {value}")

    return vector_store


def main():
    """메인 실행 함수"""
    # 환경 변수 로드
    load_dotenv()

    # 설정
    PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
    DOCS_DIR = os.path.join(PROJECT_ROOT, "src/data/safety_docs")
    OUTPUT_DIR = os.getenv("VECTOR_DB_PATH", "./vector_db")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        print("   Please set your OpenAI API key in .env:")
        print("   OPENAI_API_KEY=sk-your-api-key-here")
        return

    # 벡터 데이터베이스 구축
    try:
        vector_store = build_vector_database(
            docs_dir=DOCS_DIR,
            output_dir=OUTPUT_DIR,
            embedding_dimension=EMBEDDING_DIMENSION,
            api_key=OPENAI_API_KEY
        )

        # 간단한 검색 테스트
        print("\n" + "="*80)
        print("🧪 Quick Search Test")
        print("="*80 + "\n")

        test_queries = [
            "헬멧 안 쓰면 벌금?",
            "헬멧 착용법",
            "사고 사례"
        ]

        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            results = vector_store.search(query, top_k=2)

            for i, result in enumerate(results, 1):
                print(f"\n  [{i}] Score: {result['score']:.3f}, Distance: {result['distance']:.3f}")
                print(f"      Text: {result['text'][:100]}...")
                print(f"      Category: {result['metadata'].get('category', 'N/A')}")

        print("\n" + "="*80)
        print("✅ Vector database is ready for use!")
        print("   Run 'uv run python src/rag/query_engine.py' to test the RAG system")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error building vector database: {e}")
        raise


if __name__ == "__main__":
    main()
