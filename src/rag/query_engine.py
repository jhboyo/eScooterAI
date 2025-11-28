"""
RAG Query Engine - LLM-based Answer Generation

Vector Search + LLM Generation for domain-specific QA
FAISS 검색 결과를 컨텍스트로 사용하여 OpenAI LLM이 답변 생성
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI

from .vector_store import FAISSVectorStore


class RAGQueryEngine:
    """
    RAG (Retrieval-Augmented Generation) 쿼리 엔진

    1. Retrieval: FAISS 벡터 검색으로 관련 문서 검색
    2. Augmentation: 검색된 문서를 컨텍스트로 프롬프트 구성
    3. Generation: OpenAI LLM으로 답변 생성

    Attributes:
        vector_store (FAISSVectorStore): 벡터 저장소
        client (OpenAI): OpenAI API 클라이언트
        model (str): LLM 모델 (gpt-4-turbo-preview 또는 gpt-3.5-turbo)
        temperature (float): LLM 온도 (0.0~1.0, 낮을수록 일관적)
        max_tokens (int): 답변 최대 길이
        top_k (int): 검색할 문서 개수
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.3,
        max_tokens: int = 500,
        top_k: int = 3,
        api_key: Optional[str] = None
    ):
        """
        RAG 쿼리 엔진 초기화

        Args:
            vector_store: FAISS 벡터 저장소
            model: OpenAI LLM 모델명
            temperature: LLM 온도 (낮을수록 사실 기반, 높을수록 창의적)
            max_tokens: 답변 최대 토큰 수
            top_k: 검색할 문서 개수
            api_key: OpenAI API 키 (없으면 환경 변수)
        """
        self.vector_store = vector_store
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k

    def _build_prompt(self, query: str, contexts: List[Dict]) -> str:
        """
        프롬프트 구성 (Few-shot + Context-aware)

        검색된 문서들을 컨텍스트로 포함하여 프롬프트 생성

        Args:
            query: 사용자 질문
            contexts: 검색된 문서 리스트

        Returns:
            str: LLM 프롬프트
        """
        # 컨텍스트 문서 포맷팅
        # 검색된 Top-K 문서들을 번호와 출처와 함께 나열
        # LLM이 참조할 수 있도록 구조화된 형태로 제공
        context_text = "\n\n".join([
            f"[문서 {i+1}] (출처: {ctx['metadata'].get('source', 'N/A')})\n{ctx['text']}"
            for i, ctx in enumerate(contexts)
        ])

        # 프롬프트 구성 (Few-shot + Domain-specific)
        # RAG의 핵심: Retrieval된 문서를 컨텍스트로 제공하여 환각(Hallucination) 방지
        #
        # 프롬프트 엔지니어링 전략:
        # 1. Role-playing: "전동킥보드 안전 교육 전문가" 역할 부여
        # 2. Context grounding: 참고 문서를 명시적으로 제공
        # 3. Instruction: 5가지 명확한 지침 제시
        # 4. Few-shot (암묵적): 답변 형식 예시 내포
        # 5. Constraint: 2-3문장 제한으로 간결성 확보
        prompt = f"""당신은 전동킥보드 안전 교육 전문가입니다. 아래 참고 문서를 기반으로 질문에 답변하세요.

**참고 문서:**
{context_text}

**중요 지침:**
1. 참고 문서의 내용을 기반으로 정확하게 답변하세요.
2. 법규는 정확한 조항과 벌금을 명시하세요.
3. 안전 가이드는 구체적인 방법을 설명하세요.
4. 참고 문서에 없는 내용은 "제공된 자료에는 해당 정보가 없습니다"라고 답변하세요.
5. 답변은 2-3문장으로 간결하게 작성하세요.

**질문:** {query}

**답변:**"""

        return prompt

    def query(self, question: str, return_sources: bool = True) -> Dict[str, any]:
        """
        RAG 기반 질의응답

        1. FAISS 벡터 검색으로 관련 문서 검색 (Retrieval)
        2. 검색된 문서를 컨텍스트로 프롬프트 구성 (Augmentation)
        3. OpenAI LLM으로 답변 생성 (Generation)

        Args:
            question: 사용자 질문
            return_sources: 출처 문서 반환 여부

        Returns:
            Dict: 답변 결과
                - answer (str): 생성된 답변
                - sources (List[Dict]): 참고한 문서 리스트 (return_sources=True일 때)
                - metadata (Dict): 메타데이터 (모델, 검색 결과 수 등)

        Example:
            >>> engine = RAGQueryEngine(vector_store)
            >>> result = engine.query("헬멧 안 쓰면 벌금 얼마야?")
            >>> print(result["answer"])
            "도로교통법 제160조에 따라 헬멧 미착용 시 과태료 2만원이 부과됩니다."
        """
        # ========================================================================
        # 1. Retrieval: FAISS 벡터 검색으로 관련 문서 검색
        # ========================================================================
        # - 사용자 질문을 임베딩 벡터로 변환
        # - FAISS IndexFlatL2로 L2 거리 기반 Top-K 검색
        # - 의미적으로 유사한 문서 K개 반환 (Semantic Search)
        search_results = self.vector_store.search(question, top_k=self.top_k)

        # 검색 실패 처리 (관련 문서 없음)
        if not search_results:
            return {
                "answer": "죄송합니다. 관련 정보를 찾을 수 없습니다. 질문을 다르게 표현해주시겠어요?",
                "sources": [],
                "metadata": {
                    "model": self.model,
                    "num_sources": 0,
                    "search_success": False
                }
            }

        # ========================================================================
        # 2. Augmentation: 검색된 문서를 컨텍스트로 프롬프트 구성
        # ========================================================================
        # - 검색된 문서들을 프롬프트에 삽입
        # - LLM이 참조할 수 있는 지식 증강 (Knowledge Augmentation)
        # - 환각(Hallucination) 방지: 문서에 기반한 답변 유도
        prompt = self._build_prompt(question, search_results)

        # ========================================================================
        # 3. Generation: OpenAI LLM으로 답변 생성
        # ========================================================================
        # OpenAI Chat Completions API 호출
        # - model: gpt-4-turbo-preview 또는 gpt-3.5-turbo
        # - temperature: 0.3 (낮음, 사실 기반 답변에 적합)
        #   * 0.0: 결정론적 (항상 같은 답변)
        #   * 1.0: 창의적 (매번 다른 답변)
        # - max_tokens: 500 (답변 길이 제한)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # System message: AI의 역할 정의 (페르소나 설정)
                {"role": "system", "content": "당신은 전동킥보드 안전 교육 전문가입니다."},
                # User message: 실제 프롬프트 (질문 + 컨텍스트)
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,  # 낮은 온도 → 일관적 답변
            max_tokens=self.max_tokens  # 답변 길이 제한
        )

        # LLM 응답에서 답변 텍스트 추출
        answer = response.choices[0].message.content.strip()

        # 결과 구성
        result = {
            "answer": answer,
            "metadata": {
                "model": self.model,
                "num_sources": len(search_results),
                "search_success": True,
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        }

        # 출처 문서 포함 여부
        if return_sources:
            result["sources"] = [
                {
                    "text": doc["text"],
                    "category": doc["metadata"].get("category", "N/A"),
                    "source": doc["metadata"].get("source", "N/A"),
                    "score": doc["score"],
                    "distance": doc["distance"]
                }
                for doc in search_results
            ]

        return result

    def batch_query(self, questions: List[str]) -> List[Dict[str, any]]:
        """
        배치 질의응답

        여러 질문을 한 번에 처리 (평가용)

        Args:
            questions: 질문 리스트

        Returns:
            List[Dict]: 각 질문에 대한 답변 결과
        """
        return [self.query(q) for q in questions]

    def get_stats(self) -> Dict[str, any]:
        """
        RAG 시스템 통계

        Returns:
            Dict: 시스템 설정 및 통계
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_k": self.top_k,
            "vector_store_stats": self.vector_store.get_stats()
        }


if __name__ == "__main__":
    # 사용 예시
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # 벡터 저장소 로드 (사전에 구축되어 있어야 함)
    vector_store = FAISSVectorStore()

    try:
        vector_store.load("./vector_db")
        print("✅ 벡터 저장소 로드 완료")
    except FileNotFoundError:
        print("❌ 벡터 저장소가 없습니다. 먼저 build_vector_db.py를 실행하세요.")
        exit(1)

    # RAG 엔진 초기화
    engine = RAGQueryEngine(
        vector_store=vector_store,
        model=os.getenv("RAG_LLM_MODEL", "gpt-4-turbo-preview"),
        temperature=float(os.getenv("RAG_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("RAG_MAX_TOKENS", "500")),
        top_k=int(os.getenv("RAG_TOP_K", "3"))
    )

    print(f"\n📊 RAG 시스템 통계:")
    print(engine.get_stats())

    # 테스트 질문
    test_questions = [
        "헬멧 안 쓰면 벌금 얼마야?",
        "헬멧 올바르게 착용하는 방법 알려줘",
        "전동킥보드 타다가 사고나면 어떻게 해?",
    ]

    print("\n" + "="*80)
    print("RAG 질의응답 테스트")
    print("="*80)

    for question in test_questions:
        print(f"\n❓ 질문: {question}")
        result = engine.query(question)

        print(f"💬 답변: {result['answer']}")
        print(f"📚 참고 문서 수: {result['metadata']['num_sources']}")
        print(f"🔢 토큰 사용: {result['metadata']['total_tokens']} tokens")

        if result.get("sources"):
            print("\n📖 출처:")
            for i, source in enumerate(result["sources"], 1):
                print(f"  [{i}] {source['source']} (유사도: {source['score']:.3f})")
