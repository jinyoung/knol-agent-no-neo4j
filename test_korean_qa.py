import os
from dotenv import load_dotenv
from models import Node
from legal_doc_processor import process_document_chunk, GraphState
from qa_agent import query_knowledge_graph
import uuid

# Load environment variables
load_dotenv()

def test_korean_qa():
    """Test Korean document processing and QA."""
    print("\n=== 한국어 문서 처리 및 질의응답 테스트 시작 ===\n")
    
    # Test documents
    documents = [
        """테크놀로지 주식회사는 2020년에 설립된 AI 기술 회사입니다. 
        주요 사업 분야는 자연어 처리와 컴퓨터 비전이며, 
        현재 직원 수는 100명입니다.
        회사의 대표이사는 김철수이며, CTO는 박영희입니다.""",
        
        """회사의 주력 제품은 'AI 어시스턴트'입니다.
        이 제품은 한국어와 영어를 모두 지원하며,
        월 구독료는 10만원입니다.
        2023년 매출액은 50억원을 달성했습니다.""",
        
        """회사는 개발팀, 영업팀, 마케팅팀으로 구성되어 있습니다.
        개발팀장은 이지훈이고 30명의 개발자가 있습니다.
        영업팀장은 최수진이며 20명의 영업사원이 있습니다.
        마케팅팀장은 정민우이고 15명의 마케터가 있습니다.""",
        
        """회사는 서울 강남구 테헤란로에 위치해 있습니다.
        지하철 2호선 강남역에서 도보 5분 거리입니다.
        사무실은 지상 20층 건물의 15층에 있으며,
        총 면적은 1000평방미터입니다."""
    ]
    
    print("[문서 처리 중...]\n")
    
    # 문서 처리
    graph_state = GraphState()
    for doc in documents:
        print("처리 중인 문서:")
        print(doc)
        print()
        process_document_chunk(doc, graph_state)
        print()
    
    # 테스트 질의
    print("\n=== 질의응답 테스트 시작 ===\n")
    
    test_queries = [
        "회사의 대표이사와 CTO는 누구인가요?",
        "회사의 주력 제품과 가격은 얼마인가요?",
        "각 팀의 구성원 수는 어떻게 되나요?",
        "회사의 위치와 규모는 어떻게 되나요?",
        "2023년 회사의 매출액은 얼마였나요?",
        "회사가 설립된 연도는 언제인가요?"
    ]
    
    for query in test_queries:
        print(f"질문: {query}")
        try:
            result = query_knowledge_graph(query, graph_state.nodes)
            print(f"답변: {result['answer']}")
            print(f"신뢰도: {result['confidence']}")
            print(f"참고 노드: {result['relevant_nodes']}\n")
        except Exception as e:
            print(f"오류 발생: {str(e)}\n")

if __name__ == "__main__":
    test_korean_qa() 