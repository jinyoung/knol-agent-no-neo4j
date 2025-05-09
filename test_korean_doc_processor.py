from legal_doc_processor import Node, GraphState, process_document_chunk, create_legal_doc_processor

def test_korean_document_processing():
    """Test processing Korean text with relationships and distributed information."""
    print("\n=== 한국어 문서 처리 테스트 시작 ===")
    
    # Initialize processor
    processor = create_legal_doc_processor()
    state = GraphState()
    
    # Test chunk 1
    print("\n[처리 중인 Chunk 1]")
    print("입력: 영희는 철수와 고향친구이다.")
    chunk1 = "영희는 철수와 고향친구이다."
    state = process_document_chunk(processor, state, chunk1)
    
    print("\n현재 상태:")
    for node_id, node in state.nodes.items():
        print(f"\nNode: {node_id}")
        print(f"Title: {node.title}")
        print(f"Content: {node.content}")
        print(f"Relationships: {node.relationships}")
    
    # Test chunk 2
    print("\n[처리 중인 Chunk 2]")
    print("입력: 영희는 부산에 살고 있다. 철수와는 어떤일인지 최근에는 만나지 않는다.")
    chunk2 = "영희는 부산에 살고 있다. 철수와는 어떤일인지 최근에는 만나지 않는다."
    state = process_document_chunk(processor, state, chunk2)
    
    print("\n현재 상태:")
    for node_id, node in state.nodes.items():
        print(f"\nNode: {node_id}")
        print(f"Title: {node.title}")
        print(f"Content: {node.content}")
        print(f"Relationships: {node.relationships}")
    
    # Test chunk 3
    print("\n[처리 중인 Chunk 3]")
    print("입력: 철수는 수진이와 결혼했고, 수진이는 회를 좋아한다. 가끔 둘은 부산에 놀러가서 회를 즐긴다. 부산은 회가 맛있다.")
    chunk3 = "철수는 수진이와 결혼했고, 수진이는 회를 좋아한다. 가끔 둘은 부산에 놀러가서 회를 즐긴다. 부산은 회가 맛있다."
    state = process_document_chunk(processor, state, chunk3)
    
    print("\n현재 상태:")
    for node_id, node in state.nodes.items():
        print(f"\nNode: {node_id}")
        print(f"Title: {node.title}")
        print(f"Content: {node.content}")
        print(f"Relationships: {node.relationships}")
    
    # Test chunk 4
    print("\n[처리 중인 Chunk 4]")
    print("입력: 수진이는 서울에 철수와 함께 살고 있다. 서울은 부산에서 가기에는 거리가 멀다")
    chunk4 = "수진이는 서울에 철수와 함께 살고 있다. 서울은 부산에서 가기에는 거리가 멀다"
    state = process_document_chunk(processor, state, chunk4)
    
    print("\n=== 최종 상태 ===")
    for node_id, node in state.nodes.items():
        print(f"\nNode: {node_id}")
        print(f"Title: {node.title}")
        print(f"Content: {node.content}")
        print(f"Relationships: {node.relationships}")

if __name__ == "__main__":
    test_korean_document_processing() 