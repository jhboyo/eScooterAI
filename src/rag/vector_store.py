"""
FAISS Vector Store for RAG System

Meta AI의 FAISS (Facebook AI Similarity Search)를 사용한 벡터 유사도 검색
OpenAI Embeddings + FAISS L2 Distance 기반 의미적 문서 검색
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
from openai import OpenAI


class FAISSVectorStore:
    """
    FAISS 기반 벡터 저장소

    OpenAI text-embedding-3-small 모델로 텍스트를 벡터화하고,
    FAISS IndexFlatL2로 L2 거리 기반 유사도 검색 수행

    Attributes:
        dimension (int): 임베딩 벡터 차원 (OpenAI: 1536)
        index (faiss.Index): FAISS 인덱스
        documents (List[str]): 문서 텍스트 리스트
        metadatas (List[Dict]): 문서 메타데이터 리스트
        client (OpenAI): OpenAI API 클라이언트
    """

    def __init__(self, dimension: int = 1536, api_key: Optional[str] = None):
        """
        벡터 저장소 초기화

        Args:
            dimension: 임베딩 벡터 차원 (기본: 1536, OpenAI text-embedding-3-small)
            api_key: OpenAI API 키 (없으면 환경 변수에서 가져옴)
        """
        self.dimension = dimension

        # FAISS 인덱스 생성 (L2 거리 기반)
        self.index = faiss.IndexFlatL2(dimension)

        # 문서 및 메타데이터 저장소
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []

        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=api_key)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        텍스트를 임베딩 벡터로 변환

        OpenAI text-embedding-3-small 모델 사용
        - 차원: 1536
        - 비용: $0.00002 / 1K tokens
        - 속도: 빠름

        Args:
            text: 임베딩할 텍스트

        Returns:
            np.ndarray: 1536차원 임베딩 벡터 (float32)
        """
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = np.array(response.data[0].embedding, dtype='float32')
        return embedding

    def add_documents(self, docs: List[Dict[str, any]]) -> None:
        """
        문서를 벡터 저장소에 추가

        각 문서를 임베딩하고 FAISS 인덱스에 추가

        Args:
            docs: 문서 리스트. 각 문서는 다음 키를 가진 딕셔너리:
                - text (str): 문서 텍스트 (필수)
                - metadata (Dict): 메타데이터 (선택, 예: category, source)

        Example:
            >>> docs = [
            ...     {
            ...         "text": "헬멧 미착용 시 과태료 2만원",
            ...         "metadata": {"category": "법규", "source": "도로교통법"}
            ...     }
            ... ]
            >>> vector_store.add_documents(docs)
        """
        for doc in docs:
            text = doc["text"]
            metadata = doc.get("metadata", {})

            # 텍스트 임베딩 생성
            embedding = self.get_embedding(text)

            # FAISS 인덱스에 추가 (shape: (1, dimension))
            self.index.add(embedding.reshape(1, -1))

            # 문서 및 메타데이터 저장
            self.documents.append(text)
            self.metadatas.append(metadata)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, any]]:
        """
        쿼리와 유사한 문서 검색

        L2 거리 기반 Top-K 검색 수행
        거리가 가까울수록 유사함 (0에 가까울수록 동일)

        Args:
            query: 검색 쿼리 텍스트
            top_k: 반환할 문서 개수 (기본: 3)

        Returns:
            List[Dict]: 검색 결과 리스트. 각 결과는:
                - text (str): 문서 텍스트
                - metadata (Dict): 메타데이터
                - distance (float): L2 거리 (낮을수록 유사)
                - score (float): 유사도 점수 (높을수록 유사, 0~1)

        Example:
            >>> results = vector_store.search("헬멧 안 쓰면 벌금?", top_k=3)
            >>> print(results[0]["text"])
            "헬멧 미착용 시 과태료 2만원"
        """
        if self.index.ntotal == 0:
            return []

        # 쿼리 임베딩 생성
        query_embedding = self.get_embedding(query)

        # FAISS 검색 (L2 거리 기반, Top-K)
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            min(top_k, self.index.ntotal)  # top_k가 전체 문서 수보다 크면 조정
        )

        # 결과 구성
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:  # 유효한 인덱스
                distance = float(distances[0][i])

                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "distance": distance,
                    "score": 1.0 / (1.0 + distance)  # 거리를 점수로 변환 (0~1)
                })

        return results

    def save(self, path: str) -> None:
        """
        벡터 저장소를 디스크에 저장

        FAISS 인덱스와 문서 데이터를 별도 파일로 저장:
        - {path}/index.faiss: FAISS 인덱스
        - {path}/documents.json: 문서 및 메타데이터

        Args:
            path: 저장 디렉토리 경로
        """
        # 디렉토리 생성
        Path(path).mkdir(parents=True, exist_ok=True)

        # FAISS 인덱스 저장
        faiss.write_index(self.index, f"{path}/index.faiss")

        # 문서 및 메타데이터 저장
        with open(f"{path}/documents.json", "w", encoding="utf-8") as f:
            json.dump({
                "documents": self.documents,
                "metadatas": self.metadatas,
                "dimension": self.dimension
            }, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """
        디스크에서 벡터 저장소 로드

        Args:
            path: 저장 디렉토리 경로
        """
        # FAISS 인덱스 로드
        index_path = f"{path}/index.faiss"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")

        self.index = faiss.read_index(index_path)

        # 문서 및 메타데이터 로드
        docs_path = f"{path}/documents.json"
        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"Documents file not found at {docs_path}")

        with open(docs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.metadatas = data["metadatas"]
            self.dimension = data.get("dimension", self.dimension)

    def get_stats(self) -> Dict[str, any]:
        """
        벡터 저장소 통계 반환

        Returns:
            Dict: 통계 정보
                - total_documents: 전체 문서 수
                - dimension: 벡터 차원
                - index_type: FAISS 인덱스 타입
        """
        return {
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "index_type": "IndexFlatL2",
            "index_size": self.index.ntotal
        }


if __name__ == "__main__":
    # 사용 예시
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # 벡터 저장소 생성
    vector_store = FAISSVectorStore()

    # 샘플 문서 추가
    sample_docs = [
        {
            "text": "도로교통법 제50조 제3항: 개인형 이동장치를 운전하는 사람은 헬멧을 착용해야 합니다.",
            "metadata": {"category": "법규", "source": "도로교통법 제50조"}
        },
        {
            "text": "도로교통법 제160조: 헬멧을 착용하지 않으면 과태료 2만원이 부과됩니다.",
            "metadata": {"category": "법규", "source": "도로교통법 제160조"}
        },
        {
            "text": "헬멧은 머리 위에서 1-2cm 떨어진 위치에 착용하고, 턱끈을 단단히 조여야 합니다.",
            "metadata": {"category": "가이드", "source": "헬멧 착용법"}
        }
    ]

    print("문서 추가 중...")
    vector_store.add_documents(sample_docs)

    print(f"\n통계: {vector_store.get_stats()}")

    # 검색 테스트
    queries = [
        "헬멧 안 쓰면 벌금 얼마야?",
        "헬멧 올바르게 착용하는 방법",
    ]

    for query in queries:
        print(f"\n쿼리: {query}")
        results = vector_store.search(query, top_k=2)

        for i, result in enumerate(results, 1):
            print(f"\n결과 {i} (점수: {result['score']:.3f}, 거리: {result['distance']:.3f})")
            print(f"문서: {result['text']}")
            print(f"카테고리: {result['metadata'].get('category', 'N/A')}")

    # 저장 및 로드 테스트
    print("\n\n벡터 저장소 저장 중...")
    vector_store.save("./vector_db_test")

    print("벡터 저장소 로드 중...")
    new_store = FAISSVectorStore()
    new_store.load("./vector_db_test")

    print(f"로드된 통계: {new_store.get_stats()}")
