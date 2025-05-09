from typing import Dict, List, Any
from pydantic import BaseModel

class NodeMetadata(BaseModel):
    content_summary: Dict[str, Any] = {}
    attributes: Dict[str, Any] = {}

class NodeRelationship(BaseModel):
    node_id: str
    additional: str = ""

class Node(BaseModel):
    id: str
    title: str
    type: str = "default"
    summary: Dict[str, Any] = {}
    relationships: Dict[str, List[NodeRelationship]] = {}
    metadata: NodeMetadata = NodeMetadata() 