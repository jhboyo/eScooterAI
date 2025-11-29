"""
RAG System Testing and Evaluation

RAG 시스템 성능 평가 및 테스트
- Precision@K: Top-K 검색 정확도
- Answer Relevance: 답변 관련성
- Response Time: 응답 시간
- Hallucination Check: 환각 현상 검증
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag.vector_store import FAISSVectorStore
from src.rag.query_engine import RAGQueryEngine


class RAGTester:
    """
    RAG 시스템 테스터

    다양한 질문 유형으로 RAG 시스템 성능 평가
    """

    def __init__(self, vector_store: FAISSVectorStore, query_engine: RAGQueryEngine):
        """
        RAG 테스터 초기화

        Args:
            vector_store: FAISS 벡터 저장소
            query_engine: RAG 쿼리 엔진
        """
        self.vector_store = vector_store
        self.query_engine = query_engine

    def test_retrieval_precision(
        self,
        test_cases: List[Tuple[str, str]]
    ) -> Dict[str, any]:
        """
        검색 정확도 테스트 (Precision@K)

        Args:
            test_cases: (query, expected_category) 쌍의 리스트

        Returns:
            Dict: 평가 결과
        """
        print("\n" + "="*80)
        print("📊 Retrieval Precision@K Test")
        print("="*80 + "\n")

        # ========================================================================
        # Precision@K 평가
        # ========================================================================
        # 정보 검색(IR) 분야의 대표적인 평가 지표
        # - K개의 검색 결과 중 관련 문서가 포함된 비율
        # - Formula: Precision@K = (관련 문서 수) / K
        # - 본 테스트: Top-3 중 하나라도 올바른 카테고리면 정답 (Recall@3)
        #
        # 예시:
        # - Query: "헬멧 안 쓰면 벌금?"
        # - Expected: "법규" 카테고리
        # - Top-3 결과: ["법규", "가이드", "사례"]
        # - Hit: True (법규 포함됨)
        total = len(test_cases)
        correct = 0
        results = []

        for query, expected_category in test_cases:
            # FAISS 벡터 검색 수행 (Top-3)
            search_results = self.vector_store.search(query, top_k=3)

            # Top-3 중 하나라도 expected_category와 일치하면 정답
            # any(): 리스트에서 하나라도 True면 True 반환
            hit = any(
                doc["metadata"].get("category") == expected_category
                for doc in search_results
            )

            if hit:
                correct += 1  # 정답 카운트 증가

            results.append({
                "query": query,
                "expected_category": expected_category,
                "retrieved_categories": [
                    doc["metadata"].get("category", "N/A")
                    for doc in search_results
                ],
                "hit": hit
            })

            status = "✅" if hit else "❌"
            print(f"{status} Query: {query}")
            print(f"   Expected: {expected_category}")
            print(f"   Retrieved: {[doc['metadata'].get('category') for doc in search_results]}")

        precision = correct / total if total > 0 else 0

        print(f"\n📈 Precision@3: {precision:.2%} ({correct}/{total})")

        return {
            "precision": precision,
            "correct": correct,
            "total": total,
            "results": results
        }

    def test_answer_quality(
        self,
        test_questions: List[str]
    ) -> Dict[str, any]:
        """
        답변 품질 테스트

        Args:
            test_questions: 테스트 질문 리스트

        Returns:
            Dict: 평가 결과
        """
        print("\n" + "="*80)
        print("💬 Answer Quality Test")
        print("="*80 + "\n")

        # ========================================================================
        # 답변 품질 평가
        # ========================================================================
        # RAG 시스템의 종단간(End-to-End) 성능 평가
        # 측정 지표:
        # 1. Response Time: 답변 생성 속도 (초)
        #    - Retrieval + LLM 호출 포함
        # 2. Token Usage: OpenAI API 토큰 사용량
        #    - 비용 추정 및 최적화에 활용
        # 3. Answer Relevance: 답변 관련성 (수동 평가 필요)
        #
        # 자동화된 평가를 위해서는:
        # - GPT-4 as Judge: LLM으로 답변 품질 평가
        # - BLEU/ROUGE: 참조 답변과 비교
        # - Semantic Similarity: 임베딩 코사인 유사도
        results = []
        total_time = 0
        total_tokens = 0

        for i, question in enumerate(test_questions, 1):
            print(f"\n[{i}/{len(test_questions)}] Question: {question}")

            # 시간 측정 시작 (Retrieval + Generation 전체 시간)
            start_time = time.time()
            result = self.query_engine.query(question, return_sources=True)
            elapsed_time = time.time() - start_time

            # 누적 통계 업데이트
            total_time += elapsed_time
            total_tokens += result["metadata"]["total_tokens"]

            print(f"   Answer: {result['answer']}")
            print(f"   Response Time: {elapsed_time:.2f}s")
            print(f"   Tokens Used: {result['metadata']['total_tokens']}")
            print(f"   Sources: {result['metadata']['num_sources']}")

            results.append({
                "question": question,
                "answer": result["answer"],
                "response_time": elapsed_time,
                "tokens": result["metadata"]["total_tokens"],
                "num_sources": result["metadata"]["num_sources"],
                "sources": result.get("sources", [])
            })

        avg_time = total_time / len(test_questions) if test_questions else 0
        avg_tokens = total_tokens / len(test_questions) if test_questions else 0

        print(f"\n📈 Average Response Time: {avg_time:.2f}s")
        print(f"📈 Average Tokens Used: {avg_tokens:.1f}")

        return {
            "results": results,
            "avg_response_time": avg_time,
            "avg_tokens": avg_tokens,
            "total_questions": len(test_questions)
        }

    def test_edge_cases(self) -> Dict[str, any]:
        """
        엣지 케이스 테스트

        - 없는 정보 질문 (Hallucination 체크)
        - 모호한 질문
        - 복합 질문

        Returns:
            Dict: 평가 결과
        """
        print("\n" + "="*80)
        print("🧪 Edge Cases Test")
        print("="*80 + "\n")

        # ========================================================================
        # 엣지 케이스 평가
        # ========================================================================
        # RAG 시스템의 견고성(Robustness) 검증
        #
        # 1. Hallucination Check (환각 현상 검증)
        #    - 지식 베이스에 없는 정보를 물어봄
        #    - "모르겠다"고 답변해야 함 (거짓 정보 생성 방지)
        #    - 중요: 의료/법률 분야에서 치명적
        #
        # 2. Ambiguous Queries (모호한 질문)
        #    - 불완전하거나 애매한 질문 처리 능력
        #    - 일반적인 안전 수칙으로 답변해야 함
        #
        # 3. Complex Queries (복합 질문)
        #    - 여러 개념을 결합한 질문
        #    - 다중 문서 참조 및 추론 능력 평가
        edge_cases = [
            {
                "type": "missing_info",  # 환각 방지 테스트
                "question": "전동킥보드 보험료는 얼마야?",
                "expected_behavior": "제공된 자료에 없다고 답변"
            },
            {
                "type": "ambiguous",  # 모호한 질문 처리
                "question": "안전하게 타려면?",
                "expected_behavior": "헬멧 착용 및 안전 수칙 안내"
            },
            {
                "type": "complex",  # 복합 추론 테스트
                "question": "헬멧 안 쓰고 인도로 달리면 벌금 얼마야?",
                "expected_behavior": "헬멧 미착용(2만원) + 인도 주행(4만원) = 6만원"
            }
        ]

        results = []

        for case in edge_cases:
            print(f"\n🔍 Type: {case['type']}")
            print(f"   Question: {case['question']}")
            print(f"   Expected: {case['expected_behavior']}")

            result = self.query_engine.query(case["question"])
            print(f"   Answer: {result['answer']}")

            results.append({
                **case,
                "answer": result["answer"],
                "metadata": result["metadata"]
            })

        return {"results": results}


def main():
    """메인 테스트 실행"""
    load_dotenv()

    print("\n" + "="*80)
    print("🧪 RAG System Comprehensive Testing")
    print("="*80)

    # 1. 벡터 저장소 로드
    print("\n📂 Loading vector store...")
    vector_store = FAISSVectorStore()

    try:
        vector_store.load(os.getenv("VECTOR_DB_PATH", "./vector_db"))
        print(f"✅ Vector store loaded: {vector_store.get_stats()}")
    except FileNotFoundError:
        print("❌ Vector database not found.")
        print("   Please run 'uv run python src/rag/build_vector_db.py' first")
        return

    # 2. 쿼리 엔진 초기화
    print("\n🤖 Initializing RAG query engine...")
    query_engine = RAGQueryEngine(
        vector_store=vector_store,
        model=os.getenv("RAG_LLM_MODEL", "gpt-4-turbo-preview"),
        temperature=float(os.getenv("RAG_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("RAG_MAX_TOKENS", "500")),
        top_k=int(os.getenv("RAG_TOP_K", "3"))
    )

    print(f"✅ RAG engine ready: {query_engine.get_stats()}")

    # 3. 테스터 초기화
    tester = RAGTester(vector_store, query_engine)

    # 4. Precision@K 테스트
    precision_test_cases = [
        ("헬멧 안 쓰면 벌금?", "법규"),
        ("헬멧 착용법", "가이드"),
        ("전동킥보드 사고 사례", "사례"),
        ("음주운전 처벌", "법규"),
        ("야간 운행 주의사항", "가이드"),
        ("헬멧 착용률", "사례"),
        ("인도 주행 금지", "법규"),
        ("배터리 관리", "가이드"),
        ("2인 탑승 사고", "사례")
    ]

    precision_results = tester.test_retrieval_precision(precision_test_cases)

    # 5. 답변 품질 테스트
    quality_test_questions = [
        "헬멧 안 쓰면 벌금 얼마야?",
        "헬멧 올바르게 착용하는 방법 알려줘",
        "전동킥보드 음주운전하면 어떻게 돼?",
        "야간에 전동킥보드 타려면 뭘 켜야 해?",
        "헬멧 착용하면 사고 위험이 얼마나 줄어들어?",
        "전동킥보드 인도로 타면 안 되는 이유?",
    ]

    quality_results = tester.test_answer_quality(quality_test_questions)

    # 6. 엣지 케이스 테스트
    edge_results = tester.test_edge_cases()

    # 7. 최종 요약
    print("\n" + "="*80)
    print("📊 Final Evaluation Summary")
    print("="*80)

    print(f"\n1️⃣  Retrieval Performance:")
    print(f"   - Precision@3: {precision_results['precision']:.2%}")
    print(f"   - Correct Retrievals: {precision_results['correct']}/{precision_results['total']}")

    print(f"\n2️⃣  Answer Generation Performance:")
    print(f"   - Average Response Time: {quality_results['avg_response_time']:.2f}s")
    print(f"   - Average Tokens: {quality_results['avg_tokens']:.1f}")
    print(f"   - Total Questions Tested: {quality_results['total_questions']}")

    print(f"\n3️⃣  Edge Cases:")
    print(f"   - Total Edge Cases Tested: {len(edge_results['results'])}")

    # 비용 추정 (OpenAI Pricing 기준)
    total_tokens = quality_results['avg_tokens'] * quality_results['total_questions']
    embedding_cost = vector_store.get_stats()['total_documents'] * 0.00002 / 1000  # text-embedding-3-small
    generation_cost = total_tokens * 0.00003 / 1000  # gpt-4-turbo rough estimate

    print(f"\n💰 Estimated Cost:")
    print(f"   - Embedding Cost: ${embedding_cost:.4f}")
    print(f"   - Generation Cost: ${generation_cost:.4f}")
    print(f"   - Total: ${embedding_cost + generation_cost:.4f}")

    print("\n✅ Testing completed successfully!")


if __name__ == "__main__":
    main()
