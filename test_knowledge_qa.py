from knowledge_qa import KnowledgeGraphQA

def test_knowledge_qa():
    """Test the knowledge graph QA system."""
    print("\n=== 지식 그래프 질의응답 테스트 시작 ===")
    
    # Initialize QA system
    qa_system = KnowledgeGraphQA()
    
    # Test questions
    questions = [
        "영희와 철수의 관계는 어떻게 되나요?",
        "영희는 어디에 살고 있나요?",
        "철수와 수진이는 어떤 관계인가요?",
        "수진이는 어떤 음식을 좋아하나요?",
        "부산과 서울의 거리는 어떻게 되나요?"
    ]
    
    # Ask each question
    for question in questions:
        print(f"\n질문: {question}")
        result = qa_system.answer_question(question)
        
        print(f"답변: {result['answer']}")
        print(f"관련 노드들: {result['related_nodes']}")
        print("-" * 50)

if __name__ == "__main__":
    test_knowledge_qa() 