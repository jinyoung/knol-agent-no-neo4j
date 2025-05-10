from typing import Dict, List, Tuple, Any, Optional, Literal, TypedDict, Union
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langgraph.pregel import Graph
from llm_cache import setup_sqlite_cache
import tiktoken
from supabase import create_client, Client
import uuid
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Set up Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_KEY", "")
)

# Set up SQLite cache for LLM calls
setup_sqlite_cache()

# Configuration
MAX_TOKENS_PER_NODE = 1000  # Maximum tokens allowed in a single node before considering split
ENCODING = tiktoken.get_encoding("cl100k_base")  # GPT-4's encoding

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END

# Define data structures
class Node(BaseModel):
    """A node in the knowledge graph."""
    title: str
    content: str
    node_type: str
    relationships: Dict[str, List[Dict[str, str]]] = {}  # Changed to support richer relationship data
    metadata: Dict[str, Any] = {}
    created_at: str = ""
    updated_at: str = ""

    def __init__(self, **data):
        if "created_at" not in data:
            data["created_at"] = datetime.now().isoformat()
        if "updated_at" not in data:
            data["updated_at"] = datetime.now().isoformat()
        if "relationships" not in data:
            data["relationships"] = {}
        super().__init__(**data)
    
    def token_count(self) -> int:
        """Calculate the number of tokens in the node's content."""
        return len(ENCODING.encode(self.content))

    def would_exceed_token_limit(self, additional_content: str) -> bool:
        """Check if adding the content would exceed the token limit."""
        combined_content = self.content + "\n" + additional_content
        return len(ENCODING.encode(combined_content)) > MAX_TOKENS_PER_NODE

    def add_relationship(self, target_node_id: str, relationship_type: str, metadata: Optional[Dict[str, str]] = None):
        """Add a relationship to another node with optional metadata."""
        if relationship_type not in self.relationships:
            self.relationships[relationship_type] = []
        
        rel_data = {"node_id": target_node_id}
        if metadata:
            rel_data.update(metadata)
            
        if rel_data not in self.relationships[relationship_type]:
            self.relationships[relationship_type].append(rel_data)
            
    def merge_content(self, new_content: str, llm: Optional[ChatOpenAI] = None) -> bool:
        """Merge new content into the node, optionally using LLM for better merging."""
        if self.would_exceed_token_limit(new_content):
            return False
            
        if llm:
            merged = merge_contents(llm, self.content, new_content)
            self.content = merged
        else:
            self.content = f"{self.content}\n{new_content}"
            
        self.updated_at = datetime.now().isoformat()
        return True

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
def create_summary_chain(llm: ChatOpenAI):
    """Create a chain for summarizing content."""
    summary_prompt = ChatPromptTemplate.from_template("""
    Summarize the following content into key points and attributes.
    Return as JSON with the following structure:
    {{
        "key_points": ["point 1", "point 2", ...],
        "entities": ["entity1", "entity2", ...],
        "topics": ["topic1", "topic2", ...],
        "attributes": {{
            "attribute1": "value1",
            "attribute2": "value2"
        }}
    }}

    Content to summarize:
    {content}
    """)
    
    return (
        {"content": lambda x: x}
        | summary_prompt
        | llm
        | JsonOutputParser()
    )

def classify_chunk(state: GraphState) -> GraphState:
    """Classifies new chunk to determine relevance and relationships."""
    print("\nClassifying chunk...")
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    # Create summary chain
    summary_chain = create_summary_chain(llm)
    
    # Prepare existing nodes for context - using metadata instead of full content
    nodes_context = []
    for node_id, node in state.nodes.items():
        # If node doesn't have content summary in metadata, generate it
        if "content_summary" not in node.metadata:
            summary = summary_chain.invoke(node.content)
            node.metadata["content_summary"] = summary

        nodes_context.append({
            "id": node_id,
            "title": node.title,
            "type": node.node_type,
            "summary": node.metadata.get("content_summary", {}),
            "relationships": node.relationships
        })
    print(f"Existing nodes: {json.dumps(nodes_context, indent=2)}")
    
    # First summarize the new chunk
    chunk_summary = summary_chain.invoke(state.current_chunk)
    
    classification_prompt = ChatPromptTemplate.from_template("""
    You are an AI specialized in analyzing Korean text and organizing information into a knowledge graph.
    Your task is to analyze new information and determine how it relates to existing knowledge.
    
    Current knowledge graph nodes:
    {nodes}
    
    New text chunk summary:
    {chunk_summary}
    
    Original chunk:
    {chunk}
    
    Analyze this chunk and determine:
    1. What concepts, entities, or topics are mentioned
    2. How the information relates to existing nodes (using their summaries for comparison)
    3. Whether information should be merged into existing nodes or create new ones
    4. What relationships exist between different concepts/entities
    
    Consider these STRICT guidelines for node creation and updates:
    1. Node Consolidation Rules:
       - Each distinct concept/entity should have exactly ONE node
       - The node title should be the most concise, representative name for the concept
       - Merge all related information about a concept into its existing node
       - Only create a new node if the concept is completely new
    
    2. Content Organization Rules:
       - Each node should contain comprehensive information about its concept
       - When information connects multiple concepts, add relevant parts to each node
       - Use relationships to show how concepts are connected
       - Keep content focused and avoid redundancy
    
    3. Node Merging Priority:
       - ALWAYS check for existing nodes with similar concepts or content
       - If similar nodes exist, prefer updating existing nodes over creating new ones
       - Use clear, descriptive titles that represent the core concept
    
    Return your analysis as JSON:
    {{
        "entities": [
            {{
                "name": "concept/entity name",
                "type": "concept|process|term|principle|other",
                "node_title": "concise representative title",
                "primary_content": "main content about this concept",
                "content_summary": {{
                    "key_points": ["point 1", "point 2"],
                    "entities": ["entity1", "entity2"],
                    "topics": ["topic1", "topic2"],
                    "attributes": {{
                        "attribute1": "value1"
                    }}
                }},
                "related_content": [
                    {{
                        "target_entity": "name of related concept",
                        "content": "content describing the relationship",
                        "relationship_type": "type of relationship"
                    }}
                ]
            }}
        ],
        "node_updates": [
            {{
                "node_id": "existing node id to update",
                "title": "node title",
                "content": "content to add or merge",
                "content_summary": {{
                    "key_points": ["point 1", "point 2"],
                    "entities": ["entity1", "entity2"],
                    "topics": ["topic1", "topic2"],
                    "attributes": {{
                        "attribute1": "value1"
                    }}
                }},
                "relationships": [
                    {{
                        "target_node_id": "id of target node",
                        "relationship_type": "type of relationship",
                        "metadata": {{
                            "additional": "relationship metadata"
                        }}
                    }}
                ]
            }}
        ]
    }}
    """)
    
    classification_chain = (
        {"nodes": lambda x: json.dumps(nodes_context, indent=2),
         "chunk": lambda x: x.current_chunk,
         "chunk_summary": lambda x: json.dumps(chunk_summary, indent=2)}
        | classification_prompt
        | llm
        | JsonOutputParser()
    )
    
    state.classification = classification_chain.invoke(state)
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

def merge_contents(llm: ChatOpenAI, existing_content: str, new_content: str) -> str:
    """Merge two pieces of content using LLM to create a coherent, comprehensive description."""
    merge_prompt = ChatPromptTemplate.from_template("""
    You need to merge two pieces of information about the same topic into a single, coherent description.
    Make sure no information is lost and the result reads naturally.
    
    Existing content:
    {existing_content}
    
    New content to merge:
    {new_content}
    
    Create a single, comprehensive description that includes all the information from both sources.
    Use clear language and maintain a logical flow.
    Make sure to preserve all facts and relationships mentioned in both pieces.
    """)
    
    merge_chain = (
        {"existing_content": lambda x: x[0], "new_content": lambda x: x[1]}
        | merge_prompt
        | llm
    )
    
    result = merge_chain.invoke([existing_content, new_content])
    return result.content

# Initialize embeddings model
embeddings = OpenAIEmbeddings()

def store_node_in_supabase(node_id: str, node: Node) -> None:
    """Store a node in Supabase database.
    
    Args:
        node_id (str): The unique identifier of the node
        node (Node): The node object to store
    """
    try:
        # Generate embedding for the node content
        embedding = embeddings.embed_query(node.content)
        
        node_data = {
            "id": node_id,
            "title": node.title,
            "content": node.content,
            "node_type": node.node_type,
            "relationships": json.dumps(node.relationships),
            "metadata": json.dumps(node.metadata),
            "created_at": node.created_at,
            "updated_at": node.updated_at,
            "embedding": embedding
        }
        
        # Upsert the node data
        result = supabase.table("nodes").upsert(node_data).execute()
        
        if result.data:
            print(f"Successfully stored/updated node {node_id} in Supabase")
        else:
            print(f"Warning: No data returned when storing node {node_id}")
            
    except Exception as e:
        print(f"Error storing node {node_id} in Supabase: {str(e)}")

def implement_action(state: GraphState) -> Union[GraphState, Dict]:
    """Implements the selected strategy based on classification results."""
    print("\nImplementing action...")
    
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    summary_chain = create_summary_chain(llm)
    
    # Process new entities
    for entity in state.classification.get("entities", []):
        # Create new node
        node_id = str(uuid.uuid4())
        node = Node(
            title=entity["node_title"],
            content=entity["primary_content"],
            node_type=entity.get("type", "concept"),
            metadata={"content_summary": entity["content_summary"]}
        )
        state.nodes[node_id] = node
        
        # Process related content
        for related in entity.get("related_content", []):
            # Find or create node for related entity
            related_nodes = [
                (rid, n) for rid, n in state.nodes.items()
                if n.title == related['target_entity']
            ]
            
            if related_nodes:
                related_id, related_node = related_nodes[0]
                # Add relationship
                node.add_relationship(related_id, related["relationship_type"])
                related_node.add_relationship(node_id, related["relationship_type"])
                # Add content to related node
                if related_node.merge_content(related["content"]):
                    # Update content summary in metadata
                    summary = summary_chain.invoke(related_node.content)
                    related_node.metadata["content_summary"] = summary
            else:
                # Create new related node
                related_id = str(uuid.uuid4())
                related_node = Node(
                    title=related['target_entity'],
                    content=related["content"],
                    node_type=entity.get("type", "concept"),
                    metadata={"content_summary": {
                        "key_points": [],
                        "entities": [],
                        "topics": [],
                        "attributes": {}
                    }}
                )
                state.nodes[related_id] = related_node
                # Add relationships
                node.add_relationship(related_id, related["relationship_type"])
                related_node.add_relationship(node_id, related["relationship_type"])
                
                # Generate content summary
                summary = summary_chain.invoke(related_node.content)
                related_node.metadata["content_summary"] = summary
    
    # Process explicit node updates
    for update in state.classification.get("node_updates", []):
        if "node_id" in update and update["node_id"] in state.nodes:
            # Update existing node
            node = state.nodes[update["node_id"]]
            if node.merge_content(update["content"]):
                # Update content summary in metadata
                node.metadata["content_summary"] = update["content_summary"]
            # Update relationships
            for rel in update.get("relationships", []):
                node.add_relationship(
                    rel["target_node_id"],
                    rel["relationship_type"],
                    rel.get("metadata")
                )
    
    # After processing nodes, store them in Supabase
    for node_id, node in state.nodes.items():
        store_node_in_supabase(node_id, node)
    
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
    Analyze this content and determine how to split it into coherent sections while maintaining semantic meaning.
    Each section should be self-contained and meaningful, but try to keep related information together.
    The goal is to minimize the number of splits while keeping each section under {max_tokens} tokens.
    
    Content to analyze:
    {content}
    
    Return your analysis as JSON:
    {{
        "sections": [
            {{
                "title": "section title",
                "content": "section content that should be under {max_tokens} tokens",
                "relationships": ["relationship types with parent"]
            }}
        ],
        "common_content": "essential content that should stay in parent node (must be under {max_tokens} tokens)"
    }}
    """)
    
    # Create split chain
    split_chain = (
        {
            "content": lambda x: x.nodes[x.action_result["overflow_node"]].content,
            "max_tokens": lambda _: MAX_TOKENS_PER_NODE
        }
        | split_prompt
        | llm
        | JsonOutputParser()
    )
    
    # Run the split
    result = split_chain.invoke(state)
    
    # Update parent node with common content
    if len(ENCODING.encode(result["common_content"])) <= MAX_TOKENS_PER_NODE:
        node.content = result["common_content"]
    else:
        # If common content is still too large, keep the first part that fits
        tokens = ENCODING.encode(result["common_content"])
        decoded_content = ENCODING.decode(tokens[:MAX_TOKENS_PER_NODE])
        node.content = decoded_content
    
    # Create continuation nodes only if needed
    current_node = node
    for section in result["sections"]:
        if len(ENCODING.encode(section["content"])) > 0:  # Only create node if there's content
            section_id = f"{node_id}_continuation_{len(node.relationships.get('has_continuation', []))}"
            state.nodes[section_id] = Node(
                content=section["content"],
                node_type=node.node_type,
                relationships={"continues_from": [current_node.node_id if hasattr(current_node, 'node_id') else node_id]},
                metadata={"title": section["title"]}
            )
            
            # Update relationships
            if "has_continuation" not in current_node.relationships:
                current_node.relationships["has_continuation"] = []
            current_node.relationships["has_continuation"].append(section_id)
            current_node = state.nodes[section_id]
    
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
        nodes=state.nodes.copy(),
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
                    title=node_data.get("title", ""),
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
    
    # Merge similar nodes after processing
    result = merge_similar_nodes(result)
    
    return result

def merge_similar_nodes(state: GraphState) -> GraphState:
    """Merge similar nodes based on title and content similarity."""
    print("\nMerging similar nodes...")
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    # Update existing titles to remove 'biology' suffix
    for node_id, node in state.nodes.items():
        if "의 biology" in node.title:
            node.title = node.title.replace("의 biology", "")
    
    # Group nodes by their base title
    title_groups = {}
    for node_id, node in state.nodes.items():
        base_title = node.title.strip()
        if base_title not in title_groups:
            title_groups[base_title] = []
        title_groups[base_title].append((node_id, node))
    
    # Merge nodes with same base title
    for base_title, nodes in title_groups.items():
        if len(nodes) > 1:
            print(f"\nFound multiple nodes for '{base_title}'")
            # Use the first node as primary
            primary_id, primary_node = nodes[0]
            
            # Merge other nodes into primary
            for other_id, other_node in nodes[1:]:
                print(f"Merging node {other_id} into {primary_id}")
                # Merge content
                primary_node.merge_content(other_node.content, llm)
                
                # Merge relationships
                for rel_type, rel_list in other_node.relationships.items():
                    if rel_type not in primary_node.relationships:
                        primary_node.relationships[rel_type] = []
                    for rel in rel_list:
                        if rel not in primary_node.relationships[rel_type]:
                            primary_node.relationships[rel_type].append(rel)
                            
                # Update references to the merged node
                for _, node in state.nodes.items():
                    for rel_type, rel_list in node.relationships.items():
                        for rel in rel_list:
                            if isinstance(rel, dict) and rel.get("node_id") == other_id:
                                rel["node_id"] = primary_id
                            elif isinstance(rel, str) and rel == other_id:
                                rel_list[rel_list.index(rel)] = primary_id
                
                # Remove merged node
                del state.nodes[other_id]
    
    return state

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