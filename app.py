from flask import Flask, render_template, request, jsonify
from legal_doc_processor import create_legal_doc_processor, GraphState, Node, process_document_chunk
from qa_agent import query_knowledge_graph

app = Flask(__name__)

# Initialize the processor and state
processor = create_legal_doc_processor()
current_state = GraphState()

def convert_to_mermaid(state: GraphState) -> str:
    """Convert the knowledge graph state to Mermaid diagram format."""
    mermaid_lines = ["graph TD"]
    
    # Add nodes
    for node_id, node in state.nodes.items():
        # Create node with type
        node_label = f"{node_id}[{node_id}<br/>{node.node_type}<br/>{node.content[:50]}...]"
        mermaid_lines.append(f"    {node_label}")
        
        # Add relationships
        for rel_type, target_nodes in node.relationships.items():
            for target in target_nodes:
                # Create relationship edge with type as label
                mermaid_lines.append(f"    {node_id} -->|{rel_type}| {target}")
    
    return "\n".join(mermaid_lines)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/process_chunk', methods=['POST'])
def process_chunk():
    """Process a new document chunk and return updated graph."""
    global current_state
    
    chunk = request.json.get('chunk', '')
    if not chunk:
        return jsonify({'error': 'No chunk provided'}), 400
    
    try:
        # Process the chunk
        current_state = process_document_chunk(processor, current_state, chunk)
        
        # Convert to mermaid format
        mermaid_diagram = convert_to_mermaid(current_state)
        
        return jsonify({
            'mermaid': mermaid_diagram,
            'nodes': len(current_state.nodes)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    """Query the knowledge graph."""
    global current_state
    
    query = request.json.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        result = query_knowledge_graph(query, current_state.nodes)
        return jsonify(result)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print("\nError occurred while processing query:")
        print(error_trace)
        return jsonify({'error': f'Error processing query: {str(e)}', 'trace': error_trace}), 500

if __name__ == '__main__':
    app.run(debug=True) 