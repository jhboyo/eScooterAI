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
        # 임베딩 벡터의 차원 수 저장 (OpenAI text-embedding-3-small은 1536차원)
        self.dimension = dimension

        # FAISS 인덱스 생성 (L2 거리 기반)
        # IndexFlatL2: Brute-force L2 거리 계산을 사용하는 Flat 인덱스
        # - Flat: 모든 벡터를 순차적으로 비교 (정확도 100%)
        # - L2: 유클리드 거리 사용 (||a-b||²)
        # - 장점: 정확한 최근접 이웃 검색 (Exact NN)
        # - 단점: 대용량 데이터에서는 느림 (본 프로젝트는 35개 문서로 충분히 빠름)
        self.index = faiss.IndexFlatL2(dimension)

        # 문서 및 메타데이터 저장소
        # FAISS는 벡터만 저장하므로, 원본 텍스트와 메타데이터를 별도 리스트로 관리
        # 인덱스 i의 벡터 → documents[i], metadatas[i]로 매핑
        self.documents: List[str] = []  # 원본 텍스트 저장
        self.metadatas: List[Dict] = []  # 카테고리, 출처 등 메타데이터 저장

        # OpenAI 클라이언트 초기화
        # 임베딩 생성 및 LLM 호출에 사용
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
        # OpenAI Embeddings API 호출
        # text-embedding-3-small: 2024년 출시된 경량 임베딩 모델
        # - 1536차원 밀집 벡터 (Dense Vector) 생성
        # - Transformer 기반 인코더 아키텍처
        # - 의미적 유사도를 벡터 공간에 투영 (Semantic Embedding)
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        # API 응답에서 임베딩 벡터 추출 및 NumPy 배열로 변환
        # float32: FAISS가 요구하는 데이터 타입 (메모리 효율 + 계산 속도)
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
            # 문서에서 텍스트와 메타데이터 추출
            text = doc["text"]
            metadata = doc.get("metadata", {})

            # 텍스트 임베딩 생성 (OpenAI API 호출)
            # text → 1536차원 벡터로 변환 (의미적 표현)
            embedding = self.get_embedding(text)

            # FAISS 인덱스에 벡터 추가
            # reshape(1, -1): (1536,) → (1, 1536) 형태로 변환 (FAISS 요구사항)
            # FAISS는 2D 배열을 입력받음 (batch 처리 가능하도록)
            # add() 호출 시 내부적으로 인덱스 번호 자동 할당 (0, 1, 2, ...)
            self.index.add(embedding.reshape(1, -1))

            # 문서 및 메타데이터 저장
            # FAISS 인덱스 번호와 동일한 순서로 저장하여 매핑 유지
            # 예: FAISS 인덱스 5번 → documents[5], metadatas[5]
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
        # 빈 인덱스 체크 (문서가 없으면 검색 불가)
        if self.index.ntotal == 0:
            return []

        # 쿼리 임베딩 생성
        # 사용자 질문을 동일한 임베딩 공간으로 변환 (문서와 비교 가능하도록)
        query_embedding = self.get_embedding(query)

        # FAISS 검색 (L2 거리 기반, Top-K Nearest Neighbors)
        # 1. query_embedding과 모든 문서 벡터 간 L2 거리 계산
        #    L2 distance = ||query - doc||² = Σ(query_i - doc_i)²
        # 2. 거리가 가장 가까운 K개 문서 반환 (작을수록 유사)
        # 3. IndexFlatL2는 Exact Search (근사 아님, 정확도 100%)
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),  # (1, 1536) 형태로 변환
            min(top_k, self.index.ntotal)  # top_k가 전체 문서 수보다 크면 조정
        )
        # distances: (1, k) 형태의 거리 배열 (예: [[0.52, 0.78, 1.23]])
        # indices: (1, k) 형태의 인덱스 배열 (예: [[5, 12, 3]])

        # 결과 구성
        results = []
        for i, idx in enumerate(indices[0]):  # 첫 번째 쿼리의 결과 순회
            if idx < len(self.documents) and idx >= 0:  # 유효한 인덱스 체크
                distance = float(distances[0][i])

                results.append({
                    "text": self.documents[idx],  # 원본 텍스트
                    "metadata": self.metadatas[idx],  # 메타데이터
                    "distance": distance,  # L2 거리 (낮을수록 유사)
                    # 거리를 유사도 점수로 변환 (0~1, 높을수록 유사)
                    # score = 1 / (1 + distance)
                    # - distance=0 → score=1.0 (완전 일치)
                    # - distance=1 → score=0.5
                    # - distance=∞ → score≈0
                    "score": 1.0 / (1.0 + distance)
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
