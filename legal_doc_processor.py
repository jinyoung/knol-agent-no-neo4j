from typing import Dict, List, Tuple, Any, Optional, Literal, TypedDict, Union
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langgraph.pregel import Graph
from llm_cache import setup_sqlite_cache

# Load environment variables
load_dotenv()

# Set up SQLite cache for LLM calls
setup_sqlite_cache()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

# Define data structures
class Node(BaseModel):
    """A node in the knowledge graph."""
    content: str
    node_type: str
    relationships: Dict[str, List[str]] = {}
    metadata: Dict[str, Any] = {}
    created_at: str = ""
    updated_at: str = ""

    def __init__(self, **data):
        if "created_at" not in data:
            data["created_at"] = datetime.now().isoformat()
        if "updated_at" not in data:
            data["updated_at"] = datetime.now().isoformat()
        super().__init__(**data)

class GraphState(BaseModel):
    """State of the knowledge graph during processing."""
    nodes: Dict[str, Node] = {}
    current_chunk: str = ""
    classification: Dict[str, Any] = {}
    strategy: Dict[str, Any] = {}
    action_result: Dict[str, Any] = {}
    overflow_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary."""
        return {
            "nodes": {
                node_id: {
                    "content": node.content,
                    "node_type": node.node_type,
                    "relationships": node.relationships,
                    "metadata": node.metadata,
                    "created_at": node.created_at,
                    "updated_at": node.updated_at
                }
                for node_id, node in self.nodes.items()
            },
            "current_chunk": self.current_chunk,
            "classification": self.classification,
            "strategy": self.strategy,
            "action_result": self.action_result,
            "overflow_detected": self.overflow_detected
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphState':
        """Create a state from a dictionary."""
        nodes = {}
        for node_id, node_data in data.get("nodes", {}).items():
            nodes[node_id] = Node(
                content=node_data["content"],
                node_type=node_data["node_type"],
                relationships=node_data.get("relationships", {}),
                metadata=node_data.get("metadata", {}),
                created_at=node_data.get("created_at"),
                updated_at=node_data.get("updated_at")
            )
        
        return cls(
            nodes=nodes,
            current_chunk=data.get("current_chunk", ""),
            classification=data.get("classification", {}),
            strategy=data.get("strategy", {}),
            action_result=data.get("action_result", {}),
            overflow_detected=data.get("overflow_detected", False)
        )

# Classification Component
def classify_chunk(state: GraphState) -> GraphState:
    """Classifies new chunk to determine relevance and relationships."""
    print("\nClassifying chunk...")
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    # Prepare existing nodes for context
    nodes_context = []
    for node_id, node in state.nodes.items():
        nodes_context.append({
            "id": node_id,
            "type": node.node_type,
            "content": node.content,
            "relationships": node.relationships
        })
    print(f"Existing nodes: {json.dumps(nodes_context, indent=2)}")
    
    classification_prompt = ChatPromptTemplate.from_template("""
    You are an AI specialized in analyzing legal documents and organizing information into a knowledge graph.
    
    Current knowledge graph nodes:
    {nodes}
    
    New text chunk to analyze:
    {chunk}
    
    Analyze this chunk and determine:
    1. If it relates to existing nodes (through complementing, contradicting, or referencing them)
    2. If it contains new concepts that need new nodes
    3. What relationships exist between the information
    4. If there are any conditions or context-specific variations
    
    Return your analysis as JSON:
    {{
        "related_nodes": [
            {{
                "node_id": "id of related node",
                "relationship_type": "complements|contradicts|references|creates_exception_to",
                "conditions": "any conditions that apply to this relationship (optional)"
            }}
        ],
        "new_nodes": [
            {{
                "suggested_id": "suggested node id",
                "node_type": "concept type (e.g., regulation, exception, definition)",
                "content": "content for this node",
                "relationships": {{
                    "relationship_type": ["target_node_id"]
                }}
            }}
        ],
        "contradictions": [
            {{
                "node_id": "contradicted node id",
                "contradiction_type": "direct|conditional",
                "conditions": "conditions under which the contradiction applies (if any)",
                "resolution_strategy": "suggested strategy to resolve contradiction"
            }}
        ]
    }}
    """)
    
    # Create the classification chain
    classification_chain = (
        {"nodes": lambda x: json.dumps(nodes_context, indent=2),
         "chunk": lambda x: x.current_chunk}
        | classification_prompt
        | llm
        | JsonOutputParser()
    )
    
    # Run classification
    print("\nRunning classification...")
    result = classification_chain.invoke(state)
    print(f"Classification result: {json.dumps(result, indent=2)}")
    state.classification = result
    return state

# Strategy Selection Component
def select_strategy(state: GraphState) -> GraphState:
    """Determines how to handle the classified information."""
    print("\nSelecting strategy...")
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    strategy_prompt = ChatPromptTemplate.from_template("""
    Based on the classification results, determine the best strategy for incorporating this information.
    
    Classification results:
    {classification}
    
    Current chunk:
    {chunk}
    
    Determine the appropriate strategy:
    - "add_new": Create new nodes
    - "complement": Add information to existing nodes
    - "handle_contradiction": Resolve contradictory information
    - "create_relationships": Add new relationships between nodes
    - "split_node": Split existing node due to complexity
    
    Return your decision as JSON:
    {{
        "primary_strategy": "strategy name",
        "sub_strategies": ["additional strategies if needed"],
        "rationale": "explanation of the chosen strategy",
        "execution_order": ["ordered list of steps to take"]
    }}
    """)
    
    # Create strategy chain
    strategy_chain = (
        {"classification": lambda x: json.dumps(x.classification, indent=2),
         "chunk": lambda x: x.current_chunk}
        | strategy_prompt
        | llm
        | JsonOutputParser()
    )
    
    # Run strategy selection
    print("\nRunning strategy selection...")
    result = strategy_chain.invoke(state)
    print(f"Strategy result: {json.dumps(result, indent=2)}")
    state.strategy = result
    return state

# Action Implementation Component
def implement_action(state: GraphState) -> Union[GraphState, Dict]:
    """Implement the selected strategy on the knowledge graph."""
    print(f"\nImplementing action with strategy: {state.strategy['primary_strategy']}")
    
    if state.strategy["primary_strategy"] == "add_new":
        print("\nAdding new nodes...")
        for node in state.classification["new_nodes"]:
            node_id = node["suggested_id"]
            print(f"Creating node {node_id}")
            state.nodes[node_id] = Node(
                content=node["content"],
                node_type=node["node_type"],
                relationships=node.get("relationships", {})
            )
            
    elif state.strategy["primary_strategy"] == "complement":
        print("\nComplementing existing nodes...")
        for related_node in state.classification["related_nodes"]:
            node_id = related_node["node_id"]
            if node_id in state.nodes:
                print(f"Complementing node {node_id}")
                print(f"Existing: {state.nodes[node_id].content}")
                print(f"Adding: {related_node['conditions']}")
                state.nodes[node_id].content += f"\nAdditional context: {related_node['conditions']}"
                
    elif state.strategy["primary_strategy"] == "handle_contradiction":
        print("\nHandling contradictions...")
        for contradiction in state.classification["contradictions"]:
            node_id = contradiction["node_id"]
            if node_id not in state.nodes:
                print(f"Warning: Node {node_id} not found in state, creating new node...")
                state.nodes[node_id] = Node(
                    content="Placeholder content for contradicted node",
                    node_type="regulation",
                    relationships={}
                )
            print(f"Found {contradiction['contradiction_type']} contradiction for node {node_id}")
            state.nodes[node_id].metadata = state.nodes[node_id].metadata or {}
            state.nodes[node_id].metadata["contradiction"] = {
                "type": contradiction["contradiction_type"],
                "conditions": contradiction["conditions"],
                "resolution_strategy": contradiction["resolution_strategy"]
            }
            
    # Create relationships after handling the primary strategy
    if "create_relationships" in state.strategy.get("sub_strategies", []):
        print("\nCreating relationships...")
        for node in state.classification["new_nodes"]:
            node_id = node["suggested_id"]
            if "relationships" in node:
                for rel_type, target_nodes in node["relationships"].items():
                    print(f"Creating {rel_type} relationship from {node_id} to {target_nodes}")
                    if node_id in state.nodes:
                        state.nodes[node_id].relationships = state.nodes[node_id].relationships or {}
                        state.nodes[node_id].relationships[rel_type] = target_nodes
                        
    # Check for potential overflow in nodes
    if len(state.nodes) > 100:  # Arbitrary threshold
        print("\nWarning: Large number of nodes detected, may need to manage context...")
        state = manage_context(state)
        
    return state

# Context Management Component
def manage_context(state: GraphState) -> GraphState:
    """Handles context overflow by splitting nodes when necessary."""
    if not state.overflow_detected:
        return state
    
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    node_id = state.action_result["overflow_node"]
    node = state.nodes[node_id]
    
    split_prompt = ChatPromptTemplate.from_template("""
    Analyze this content and determine the best way to split it into coherent sub-sections:
    
    Content:
    {content}
    
    Return your analysis as JSON:
    {{
        "split_type": "categorical|sequential",
        "sections": [
            {{
                "title": "section title",
                "content": "section content",
                "relationships": ["relationship types with parent"]
            }}
        ],
        "common_content": "content that should stay in parent node"
    }}
    """)
    
    # Create split chain
    split_chain = (
        {"content": lambda x: x.nodes[x.action_result["overflow_node"]].content}
        | split_prompt
        | llm
        | JsonOutputParser()
    )
    
    # Run the split
    result = split_chain.invoke(state)
    
    # Update parent node
    node.content = result["common_content"]
    node.relationships["has_section"] = []
    
    # Create child nodes
    for i, section in enumerate(result["sections"]):
        section_id = f"{node_id}_section_{i+1}"
        state.nodes[section_id] = Node(
            content=section["content"],
            node_type=f"{node.node_type}_{section['title'].lower().replace(' ', '_')}",
            relationships={"section_of": [node_id]},
            metadata={"title": section["title"]}
        )
        node.relationships["has_section"].append(section_id)
    
    state.overflow_detected = False
    state.action_result["context_managed"] = True
    
    return state

# User Input Handler
def handle_user_input(state: GraphState, user_input: str) -> GraphState:
    """Processes user input for contradiction resolution."""
    if state.strategy.get("contradiction_query") and "contradiction" in state.strategy["contradiction_query"].lower():
        node_id = state.action_result.get("node_id")
        if node_id:
            if "conditions:" in user_input.lower():
                # User provided conditions - create polymorphic nodes
                conditions = user_input.split("conditions:")[1].strip()
                state.strategy['contradictions'] = [{
                    "node_id": node_id,
                    "contradiction_type": "conditional",
                    "conditions": conditions
                }]
                state.strategy['primary_strategy'] = "handle_contradiction"
                state = implement_action(state)
            else:
                # User provided direct resolution - update node
                state.nodes[node_id].content = user_input
                state.nodes[node_id].updated_at = datetime.now().isoformat()
            
            state.strategy["contradiction_query"] = None
    
    return state

# Define the routing logic
def should_continue(state: GraphState) -> Literal["user_input", "continue", "end"]:
    """Determines the next step in the workflow."""
    if state.strategy.get("contradiction_query"):
        return "user_input"
    elif state.overflow_detected:
        return "continue"
    return "end"

# Create the LangGraph
def create_legal_doc_processor() -> Any:
    """Creates the legal document processing workflow."""
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("classify", classify_chunk)
    workflow.add_node("select_strategy", select_strategy)
    workflow.add_node("implement_action", implement_action)
    workflow.add_node("manage_context", manage_context)
    workflow.add_node("handle_user_input", RunnablePassthrough())
    
    # Add edges starting from classify
    workflow.add_edge("classify", "select_strategy")
    workflow.add_edge("select_strategy", "implement_action")
    
    # Set the entry point
    workflow.set_entry_point("classify")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "implement_action",
        should_continue,
        {
            "user_input": "handle_user_input",
            "continue": "manage_context",
            "end": END
        }
    )
    
    workflow.add_edge("handle_user_input", "implement_action")
    workflow.add_edge("manage_context", END)
    
    # Compile the graph
    return workflow.compile()

# Example usage
def process_document_chunk(processor: Graph, state: GraphState, chunk: str) -> GraphState:
    """Process a single chunk of the document."""
    # Create a new state with the current chunk
    current_state = GraphState(
        nodes=state.nodes.copy(),  # Make a copy of the nodes dictionary
        current_chunk=chunk
    )
    
    # Invoke the processor with the current state
    result = processor.invoke(current_state)
    
    # If result is not a GraphState, convert it
    if not isinstance(result, GraphState):
        # Create a new nodes dictionary
        nodes = {}
        # Convert each node in the result
        for node_id, node_data in result.get("nodes", {}).items():
            if isinstance(node_data, Node):
                nodes[node_id] = node_data
            else:
                nodes[node_id] = Node(
                    content=node_data.get("content", ""),
                    node_type=node_data.get("node_type", "unknown"),
                    relationships=node_data.get("relationships", {}),
                    metadata=node_data.get("metadata", {}),
                    created_at=node_data.get("created_at", datetime.now().isoformat()),
                    updated_at=node_data.get("updated_at", datetime.now().isoformat())
                )
        
        # Create a new state with the converted nodes
        result = GraphState(
            nodes=nodes,
            current_chunk=result.get("current_chunk", chunk),
            classification=result.get("classification", {}),
            strategy=result.get("strategy", {}),
            action_result=result.get("action_result", {}),
            overflow_detected=result.get("overflow_detected", False)
        )
    
    return result

if __name__ == "__main__":
    # Create the processor
    processor = create_legal_doc_processor()
    
    # Initialize state with example nodes
    initial_state = GraphState(
        nodes={
            "node_1": Node(
                content="Corporate tax deductions are allowable for ordinary and necessary business expenses incurred during the taxable year.",
                node_type="tax_regulation"
            )
        }
    )
    
    # Example chunk to process
    example_chunk = """
    Business expenses must be both ordinary and necessary to qualify for tax deductions. 
    An ordinary expense is one that is common and accepted in the industry. 
    A necessary expense is one that is helpful and appropriate for the business. 
    The expense must be directly related to the business and not be lavish or extravagant under the circumstances.
    """
    
    # Process the chunk
    result_state = process_document_chunk(processor, initial_state, example_chunk)
    
    # Print results
    print("\nProcessed Knowledge Graph:")
    for node_id, node in result_state.nodes.items():
        print(f"\nNode {node_id} ({node.node_type}):")
        print(f"Content: {node.content}")
        print("Relationships:", node.relationships) 