from typing import Dict, List, Any, Optional
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage, SystemMessage
from models import Node
import json
import re
import uuid

# Load environment variables
load_dotenv()

# Set up Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_KEY", "")
)

def query_knowledge_graph(query: str, nodes: Dict[str, Node]) -> Dict[str, Any]:
    """
    Query the knowledge graph with a given question and return the answer.
    
    Args:
        query: The question to ask
        nodes: Dictionary of nodes in the knowledge graph
        
    Returns:
        Dictionary containing:
        - answer: The generated answer
        - confidence: Confidence score (0-1)
        - relevant_nodes: List of nodes used to generate the answer
        
    Raises:
        Exception: If any error occurs during processing
    """
    
    chat = ChatOpenAI(temperature=0)
    
    system_prompt = """당신은 지식 그래프를 기반으로 질문에 답변하는 AI 어시스턴트입니다.
주어진 노드들의 정보를 분석하여 질문에 가장 적절한 답변을 생성해주세요.

중요: 응답은 반드시 다음과 같은 JSON 형식이어야 합니다. 다른 형식은 허용되지 않습니다:

{
    "answer": "답변 내용 (정보가 없는 경우: '주어진 정보에서 답변을 찾을 수 없습니다')",
    "confidence": 신뢰도 점수 (0.0 ~ 1.0),
    "relevant_nodes": ["참고한 노드의 제목"]
}

규칙:
1. 응답은 반드시 위의 JSON 형식을 따라야 합니다
2. 일반 텍스트 응답은 허용되지 않습니다
3. 정보가 부족하거나 없는 경우에도 JSON 형식을 유지해야 합니다
4. 모든 답변은 한국어로 작성되어야 합니다
5. 신뢰도 점수는 0.0에서 1.0 사이의 값이어야 합니다
6. 답변에 사용된 모든 정보는 반드시 주어진 노드들의 내용에 기반해야 합니다"""

    # Convert nodes to a list of dictionaries
    nodes_data = []
    for node_id, node in nodes.items():
        node_data = {
            'id': node_id,
            'title': node.title if hasattr(node, 'title') else node.content[:50],
            'type': node.node_type,
            'content': node.content,
            'relationships': node.relationships,
            'metadata': node.metadata
        }
        nodes_data.append(node_data)

    human_prompt = f"""질문: {query}

사용 가능한 노드들:
{json.dumps(nodes_data, ensure_ascii=False, indent=2)}

주의: 응답은 반드시 지정된 JSON 형식이어야 합니다."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    # Let exceptions propagate up
    response = chat.invoke(messages)
    content = response.content
    
    print("\nRaw response from OpenAI:")
    print(content)
    print("\nAfter removing JSON tags:")
    
    # Remove any ```json or ``` tags
    content = re.sub(r'```json\s*|\s*```', '', content)
    print(content)
    
    print("\nAttempting to parse JSON...")
    
    # If response is not in JSON format, create a default JSON response
    if not content.strip().startswith('{'):
        content = json.dumps({
            "answer": "주어진 정보에서 답변을 찾을 수 없습니다",
            "confidence": 0.0,
            "relevant_nodes": []
        }, ensure_ascii=False)
        print("\nConverted non-JSON response to default JSON format")
    
    # Parse JSON response
    result = json.loads(content)
    
    # Validate required fields
    if not all(k in result for k in ['answer', 'confidence', 'relevant_nodes']):
        raise ValueError("Missing required fields in response")
        
    # Convert node titles to strings
    result['relevant_nodes'] = [str(title) for title in result['relevant_nodes']]
        
    return result 