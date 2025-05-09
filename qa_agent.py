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

def query_knowledge_graph(query: str, nodes: List[Node]) -> Dict[str, Any]:
    """
    Query the knowledge graph with a given question and return the answer.
    
    Args:
        query: The question to ask
        nodes: List of nodes in the knowledge graph
        
    Returns:
        Dictionary containing:
        - answer: The generated answer
        - confidence: Confidence score (0-1)
        - relevant_nodes: List of nodes used to generate the answer
    """
    
    chat = ChatOpenAI(temperature=0)
    
    system_prompt = """당신은 지식 그래프를 기반으로 질문에 답변하는 AI 어시스턴트입니다.
주어진 노드들의 정보를 분석하여 질문에 가장 적절한 답변을 생성해주세요.
답변은 반드시 한국어로 작성해야 하며, 다음 JSON 형식으로 반환해야 합니다:

{
    "answer": "답변 내용",
    "confidence": 신뢰도 점수 (0.0 ~ 1.0),
    "relevant_nodes": ["참고한 노드의 제목"]
}

신뢰도 점수는 답변의 정확성과 완성도를 기반으로 0.0에서 1.0 사이의 값으로 설정해주세요.
답변에 사용된 모든 정보는 반드시 주어진 노드들의 내용에 기반해야 합니다.

노드 ID는 UUID 형식이므로, 답변에서는 노드의 제목을 사용해주세요."""

    # Convert nodes to a list of dictionaries
    nodes_data = []
    for node in nodes:
        node_data = {
            'id': node.id,  # Now using UUID-based ID
            'title': node.title,
            'type': node.type,
            'summary': node.summary
        }
        
        # Handle relationships with UUID-based IDs
        relationships = {}
        for rel_type, rel_list in node.relationships.items():
            relationships[rel_type] = []
            for rel in rel_list:
                rel_data = {
                    'node_id': rel.node_id,  # UUID-based ID
                    'additional': rel.additional if rel.additional else ''
                }
                relationships[rel_type].append(rel_data)
        node_data['relationships'] = relationships
        
        nodes_data.append(node_data)

    human_prompt = f"""질문: {query}

사용 가능한 노드들:
{json.dumps(nodes_data, ensure_ascii=False, indent=2)}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    try:
        response = chat.invoke(messages)
        content = response.content

        # Remove any ```json or ``` tags
        content = re.sub(r'```json\s*|\s*```', '', content)
        
        # Parse JSON response
        result = json.loads(content)
        
        # Validate required fields
        if not all(k in result for k in ['answer', 'confidence', 'relevant_nodes']):
            raise ValueError("Missing required fields in response")
            
        # Convert node titles to strings
        result['relevant_nodes'] = [str(title) for title in result['relevant_nodes']]
            
        return result
        
    except json.JSONDecodeError as e:
        return {
            "answer": "죄송합니다. 응답을 처리하는 중에 오류가 발생했습니다.",
            "confidence": 0.0,
            "relevant_nodes": []
        }
    except Exception as e:
        return {
            "answer": f"오류 발생: {str(e)}",
            "confidence": 0.0,
            "relevant_nodes": []
        } 