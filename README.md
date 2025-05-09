# Legal Document Knowledge Graph Visualizer

This web application processes legal document chunks and visualizes the knowledge graph using Mermaid diagrams. It helps in understanding relationships and connections between different parts of legal documents.

## Features

- Process text chunks from legal documents
- Automatically extract relationships and concepts
- Visualize the knowledge graph in real-time
- Interactive web interface
- Mermaid-based graph visualization
- Persistent storage with Supabase

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_api_key_here
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

3. Set up Supabase:
   - Create a new project at [Supabase](https://supabase.com)
   - Get your project URL and anon key from the project settings
   - Run the SQL script in `scripts/create_tables.sql` in the Supabase SQL editor

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Enter a chunk of legal text in the input box
2. Click "Process Chunk" to add it to the knowledge graph
3. The graph will update automatically to show new nodes and relationships
4. Continue adding chunks to build a comprehensive knowledge graph
5. All nodes are automatically persisted to Supabase for future reference

## Technical Details

- Built with Flask
- Uses OpenAI's GPT-4 for text analysis
- Mermaid.js for graph visualization
- Tailwind CSS for styling
- Supabase for persistent storage

## Requirements

- Python 3.8+
- OpenAI API key
- Supabase account and project
- Modern web browser

## How It Works

1. **Classification**: Each chunk is analyzed to determine:
   - Relationships with existing nodes
   - Need for new nodes
   - Potential contradictions
   - Conditional variations

2. **Strategy Selection**: The agent chooses how to handle the information:
   - Add new nodes
   - Complement existing information
   - Handle contradictions
   - Create relationships
   - Split nodes due to complexity

3. **Action Implementation**: The chosen strategy is executed:
   - Creating/updating nodes
   - Managing relationships
   - Handling contradictions with polymorphic nodes
   - Splitting overflow content
   - Persisting nodes to Supabase

4. **Context Management**: Large nodes are automatically split into coherent subsections while maintaining relationships.

## Database Schema

The application uses the following Supabase table structure:

### Nodes Table
- `id`: Text (Primary Key) - Unique identifier for the node
- `title`: Text - Node title
- `content`: Text - Node content
- `node_type`: Text - Type of the node
- `relationships`: JSONB - JSON object containing relationships to other nodes
- `metadata`: JSONB - Additional node metadata
- `created_at`: Timestamp - Creation timestamp
- `updated_at`: Timestamp - Last update timestamp

## Example Output

The test file demonstrates processing tax regulation documents, showing how the knowledge graph evolves with each chunk. The output includes:

- Node content and types
- Relationships between nodes
- Metadata including conditions and timestamps
- Polymorphic variations for conditional information

## Customization

You can customize the agent by:

1. Modifying the classification prompts in `classify_chunk()`
2. Adjusting the content overflow threshold (default: 2000 characters)
3. Adding new merge strategies in `implement_action()`
4. Customizing the node splitting logic in `manage_context()`
5. Extending the Supabase schema for additional data storage

## Contributing

Feel free to submit issues and pull requests for:

- Additional merge strategies
- Improved classification logic
- Better content splitting algorithms
- New relationship types
- Documentation improvements
- Database schema enhancements

## License

MIT License 






# implementation

I'll help you set up the .env file and modify the code to use it. First, let's add python-dotenv to our requirements:

```pip requirements:requirements.txt
langgraph>=0.2.60
langchain>=0.1.0
langchain-openai>=0.0.5
pydantic>=2.0.0
openai>=1.0.0
python-dotenv>=1.0.0
```
Now, let's create a .env file with your OpenAI API key:

Now, let's modify the legal_doc_processor.py to use the .env file:

```python:legal_doc_processor.py
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
// ... existing code ...
```
Now let's run the test:

```bash
python test_legal_doc_processor.py
```
I see the error - we need to add a START node to our graph. Let's modify the create_legal_doc_processor function:

```python:legal_doc_processor.py
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
    
    # Add edges
    workflow.add_edge("START", "classify")  # Add START node
    workflow.add_edge("classify", "select_strategy")
    workflow.add_edge("select_strategy", "implement_action")
    
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

# ... rest of the code ...
```
Now let's try running the test again:

```bash
python test_legal_doc_processor.py
```
I apologize for the confusion. We need to use the correct constant for the START node. Let's modify the code:

```python:legal_doc_processor.py
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

# ... rest of the code ...
```
I've made a few key changes to fix the issue:

1. Removed the incorrect `workflow.add_edge("START", "classify")` line
2. Added `workflow.set_entry_point("classify")` which is the proper way to set the starting node in LangGraph

This change uses the correct method to set up the entry point of the graph. Instead of manually adding an edge from "START", we use the `set_entry_point()` method which properly configures the starting node.

Let's run the test again to see if this resolves the issue:

```bash
python test_legal_doc_processor.py
```
Now we're seeing a different error. The issue is in the `classify_chunk` function where there's a missing variable in the prompt template. The error indicates that the prompt is expecting a variable called "related_nodes" but it's not being provided.

Let's examine the `classify_chunk` function:

Read file: legal_doc_processor.py
Let me also check the classification prompt:

Read file: legal_doc_processor.py
I see the issue now. The error occurs because there's a formatting issue in the prompt template. The `related_nodes` variable is being interpreted as a template variable when it's actually part of the JSON schema example. Let's fix this by escaping the curly braces in the JSON example:

```python:legal_doc_processor.py
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
```
I've modified the classification prompt template by escaping all the curly braces in the JSON example. This prevents the template engine from interpreting them as variables. Now let's run the test again:

```bash
python test_legal_doc_processor.py
```
Now we're seeing a new error in the `select_strategy` function. The error occurs because we're trying to access `x["classification_result"]` but `x` is a `GraphState` object, which doesn't support dictionary-style access.

Let's fix the `select_strategy` function to properly access the state object's attributes:

```python:legal_doc_processor.py
    # Create strategy chain
    strategy_chain = (
        {"classification": lambda x: json.dumps(x.classification_result, indent=2),
         "chunk": lambda x: x.current_chunk}
        | strategy_prompt
        | llm
        | JsonOutputParser()
    )
```
I've modified the `strategy_chain` to use attribute access (`x.classification_result` and `x.current_chunk`) instead of dictionary-style access. This is the correct way to access attributes of the `GraphState` class.

Let's run the test again:

```bash
python test_legal_doc_processor.py
```
We have another template issue in the `strategy_prompt`. Let's fix that by escaping the curly braces in the JSON example:

```python:legal_doc_processor.py
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
```
I've modified the `strategy_prompt` template by escaping the curly braces in the JSON example. Let's run the test again:

```bash
python test_legal_doc_processor.py
```
Now we're seeing a different error. The issue is that the state object returned from the processor is not the same type as our `GraphState` class. Let's check the `process_document_chunk` function and the test file:

Read file: legal_doc_processor.py

Read file: test_legal_doc_processor.py
I see the issue now. The state object returned by the LangGraph processor is not being properly converted back to our `GraphState` class. We need to modify the `process_document_chunk` function to handle this conversion:

```python:legal_doc_processor.py
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
```
I've modified the `process_document_chunk` function to properly handle the conversion of the returned state object back to our `GraphState` class. The function now:

1. Checks if the result is already a `GraphState` instance
2. If not, creates a new `GraphState` instance with data from the result object
3. Uses `hasattr` checks to safely access attributes that may or may not exist
4. Provides default values for missing attributes

Let's run the test again:

```bash
python test_legal_doc_processor.py
```
Great! The test is now running without errors. However, I notice that we're not seeing any nodes in the output. This suggests that the nodes are not being properly updated in the state. Let's check the `implement_action` function since that's where the nodes should be modified:

Read file: legal_doc_processor.py
I see several issues in the `implement_action` function:

1. In the `complement` case, we're trying to access `x["nodes"]` but we should use `x.nodes`
2. In the `handle_contradiction` case, we're using dictionary-style access for state attributes
3. In the `manage_context` function, we're also using dictionary-style access

Let's fix these issues:

```python:legal_doc_processor.py
def implement_action(state: GraphState) -> GraphState:
    """Implements the selected strategy to update the knowledge graph."""
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    if state.merge_strategy == "add_new":
        # Create new nodes
        for new_node in state.classification_result["new_nodes"]:
            node_id = new_node["suggested_id"]
            state.nodes[node_id] = Node(
                content=new_node["content"],
                node_type=new_node["node_type"],
                relationships=new_node["relationships"]
            )
            
    elif state.merge_strategy == "complement":
        # Update existing nodes with new information
        for related in state.classification_result["related_nodes"]:
            if related["relationship_type"] == "complements":
                node_id = related["node_id"]
                if node_id in state.nodes:
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
        for contradiction in state.classification_result["contradictions"]:
            node_id = contradiction["node_id"]
            if contradiction["contradiction_type"] == "conditional":
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
                # Direct contradiction - need user input
                state.user_query = f"""
                Found contradictory information:
                
                Existing: {state.nodes[node_id].content}
                
                New: {state.current_chunk}
                
                Please specify which version should be kept or provide conditions under which each version applies.
                """
                return state
    
    elif state.merge_strategy == "create_relationships":
        # Add new relationships between nodes
        for related in state.classification_result["related_nodes"]:
            from_node = related.get("from_node")
            to_node = related.get("to_node")
            rel_type = related.get("relationship_type")
            
            if from_node in state.nodes and to_node in state.nodes:
                if rel_type not in state.nodes[from_node].relationships:
                    state.nodes[from_node].relationships[rel_type] = []
                state.nodes[from_node].relationships[rel_type].append(to_node)
    
    # Check for potential overflow
    for node_id, node in state.nodes.items():
        if len(node.content) > 2000:  # Arbitrary threshold
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
```
I've made several changes:

1. Fixed all instances of dictionary-style access (`x["nodes"]`) to use attribute access (`x.nodes`)
2. Escaped curly braces in the JSON template in the `split_prompt`
3. Fixed the chain access patterns in all the prompts to use attribute access

Let's run the test again:

```bash
python test_legal_doc_processor.py
```
We're still not seeing any nodes in the output. Let's check if we need to add OpenAI API key configuration since we're using the ChatOpenAI model:

```bash
echo $OPENAI_API_KEY
```
The API key is set. Let's add some logging to see what's happening in the `classify_chunk` function:

```python:legal_doc_processor.py
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
```
I've added logging to the `classify_chunk` function and also fixed a bug where we were creating a new dictionary instead of using the state object. Let's also add logging to the `select_strategy` and `implement_action` functions:

```python:legal_doc_processor.py
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
```



