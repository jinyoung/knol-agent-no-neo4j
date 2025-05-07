from typing import Dict, List, Tuple, Any, Optional, Literal, TypedDict
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Define data structures
class Node(BaseModel):
    content: str
    node_type: str
    relationships: Dict[str, List[str]] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class GraphState(BaseModel):
    nodes: Dict[str, Node] = Field(default_factory=dict)
    current_chunk: str = ""
    classification_result: Dict = Field(default_factory=dict)
    merge_strategy: str = ""
    action_result: Dict = Field(default_factory=dict)
    user_query: Optional[str] = None
    overflow_detected: bool = False
    context: Dict[str, Any] = Field(default_factory=dict)

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
    state.classification_result = result
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
        {"classification": lambda x: json.dumps(x.classification_result, indent=2),
         "chunk": lambda x: x.current_chunk}
        | strategy_prompt
        | llm
        | JsonOutputParser()
    )
    
    # Run strategy selection
    print("\nRunning strategy selection...")
    result = strategy_chain.invoke(state)
    print(f"Strategy result: {json.dumps(result, indent=2)}")
    state.merge_strategy = result["primary_strategy"]
    state.context["strategy_details"] = result
    return state

# Action Implementation Component
def implement_action(state: GraphState) -> GraphState:
    """Implements the selected strategy to update the knowledge graph."""
    print(f"\nImplementing action with strategy: {state.merge_strategy}")
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    if state.merge_strategy == "add_new":
        print("\nAdding new nodes...")
        # Create new nodes
        for new_node in state.classification_result["new_nodes"]:
            node_id = new_node["suggested_id"]
            print(f"Creating node {node_id}")
            state.nodes[node_id] = Node(
                content=new_node["content"],
                node_type=new_node["node_type"],
                relationships=new_node["relationships"]
            )
            
    elif state.merge_strategy == "complement":
        print("\nComplementing existing nodes...")
        # Update existing nodes with new information
        for related in state.classification_result["related_nodes"]:
            if related["relationship_type"] == "complements":
                node_id = related["node_id"]
                if node_id in state.nodes:
                    print(f"Updating node {node_id}")
                    merge_prompt = ChatPromptTemplate.from_template("""
                    Merge the following information coherently:
                    
                    Existing content:
                    {existing}
                    
                    New information:
                    {new_info}
                    
                    Return the merged content that preserves all important details.
                    """)
                    
                    merge_chain = (
                        {"existing": lambda x: x.nodes[node_id].content,
                         "new_info": lambda x: x.current_chunk}
                        | merge_prompt
                        | llm
                    )
                    
                    result = merge_chain.invoke(state)
                    state.nodes[node_id].content = result.content
                    state.nodes[node_id].updated_at = datetime.now().isoformat()
                    
    elif state.merge_strategy == "handle_contradiction":
        print("\nHandling contradictions...")
        for contradiction in state.classification_result["contradictions"]:
            node_id = contradiction["node_id"]
            if contradiction["contradiction_type"] == "conditional":
                print(f"Handling conditional contradiction for node {node_id}")
                # Handle conditional contradiction by creating polymorphic nodes
                original_node = state.nodes[node_id]
                
                # Create a parent node with common information
                parent_prompt = ChatPromptTemplate.from_template("""
                Extract the common, non-contradictory information from:
                
                Original content:
                {original}
                
                New contradictory content:
                {new_content}
                
                Conditions:
                {conditions}
                
                Return only the common information that applies in all cases.
                """)
                
                parent_chain = (
                    {"original": lambda x: original_node.content,
                     "new_content": lambda x: x.current_chunk,
                     "conditions": lambda x: contradiction["conditions"]}
                    | parent_prompt
                    | llm
                )
                
                parent_result = parent_chain.invoke(state)
                
                # Update original node to be the parent
                original_node.content = parent_result.content
                original_node.relationships["has_variant"] = []
                
                # Create variant nodes
                variant_prompt = ChatPromptTemplate.from_template("""
                Create specific content for a variant node based on the conditions.
                
                Conditions:
                {conditions}
                
                Content to adapt:
                {content}
                
                Return the content specifically applicable under these conditions.
                """)
                
                # Create variant for original case
                original_variant_id = f"{node_id}_variant_1"
                print(f"Creating variant node {original_variant_id}")
                variant_chain = (
                    {"conditions": lambda x: "Original conditions",
                     "content": lambda x: original_node.content}
                    | variant_prompt
                    | llm
                )
                variant_result = variant_chain.invoke(state)
                
                state.nodes[original_variant_id] = Node(
                    content=variant_result.content,
                    node_type=f"{original_node.node_type}_variant",
                    relationships={"variant_of": [node_id]},
                    metadata={"conditions": "Original conditions"}
                )
                original_node.relationships["has_variant"].append(original_variant_id)
                
                # Create variant for new case
                new_variant_id = f"{node_id}_variant_2"
                print(f"Creating variant node {new_variant_id}")
                variant_chain = (
                    {"conditions": lambda x: contradiction["conditions"],
                     "content": lambda x: x.current_chunk}
                    | variant_prompt
                    | llm
                )
                variant_result = variant_chain.invoke(state)
                
                state.nodes[new_variant_id] = Node(
                    content=variant_result.content,
                    node_type=f"{original_node.node_type}_variant",
                    relationships={"variant_of": [node_id]},
                    metadata={"conditions": contradiction["conditions"]}
                )
                original_node.relationships["has_variant"].append(new_variant_id)
                
            else:
                print(f"Found direct contradiction for node {node_id}")
                # Direct contradiction - need user input
                state.user_query = f"""
                Found contradictory information:
                
                Existing: {state.nodes[node_id].content}
                
                New: {state.current_chunk}
                
                Please specify which version should be kept or provide conditions under which each version applies.
                """
                return state
    
    elif state.merge_strategy == "create_relationships":
        print("\nCreating relationships...")
        # Add new relationships between nodes
        for related in state.classification_result["related_nodes"]:
            from_node = related.get("from_node")
            to_node = related.get("to_node")
            rel_type = related.get("relationship_type")
            
            if from_node in state.nodes and to_node in state.nodes:
                print(f"Adding relationship {rel_type} from {from_node} to {to_node}")
                if rel_type not in state.nodes[from_node].relationships:
                    state.nodes[from_node].relationships[rel_type] = []
                state.nodes[from_node].relationships[rel_type].append(to_node)
    
    # Check for potential overflow
    for node_id, node in state.nodes.items():
        if len(node.content) > 2000:  # Arbitrary threshold
            print(f"\nOverflow detected in node {node_id}")
            state.overflow_detected = True
            state.action_result["overflow_node"] = node_id
            break
    
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
    if state.user_query and "contradiction" in state.user_query.lower():
        node_id = state.action_result.get("node_id")
        if node_id:
            if "conditions:" in user_input.lower():
                # User provided conditions - create polymorphic nodes
                conditions = user_input.split("conditions:")[1].strip()
                state.classification_result["contradictions"] = [{
                    "node_id": node_id,
                    "contradiction_type": "conditional",
                    "conditions": conditions
                }]
                state.merge_strategy = "handle_contradiction"
                state = implement_action(state)
            else:
                # User provided direct resolution - update node
                state.nodes[node_id].content = user_input
                state.nodes[node_id].updated_at = datetime.now().isoformat()
            
            state.user_query = None
    
    return state

# Define the routing logic
def should_continue(state: GraphState) -> Literal["user_input", "continue", "end"]:
    """Determines the next step in the workflow."""
    if state.user_query:
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
def process_document_chunk(processor: Any, state: GraphState, chunk: str) -> GraphState:
    """Process a single chunk of the document."""
    state.current_chunk = chunk
    result = processor.invoke(state)
    
    # Convert the result back to GraphState
    if not isinstance(result, GraphState):
        # Create a new GraphState with the data from result
        new_state = GraphState(
            nodes=result.nodes if hasattr(result, 'nodes') else {},
            current_chunk=result.current_chunk if hasattr(result, 'current_chunk') else "",
            classification_result=result.classification_result if hasattr(result, 'classification_result') else {},
            merge_strategy=result.merge_strategy if hasattr(result, 'merge_strategy') else "",
            action_result=result.action_result if hasattr(result, 'action_result') else {},
            user_query=result.user_query if hasattr(result, 'user_query') else None,
            overflow_detected=result.overflow_detected if hasattr(result, 'overflow_detected') else False,
            context=result.context if hasattr(result, 'context') else {}
        )
        return new_state
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