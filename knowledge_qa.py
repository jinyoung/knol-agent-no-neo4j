from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# Set up Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL", ""),
    os.getenv("SUPABASE_KEY", "")
)

class KnowledgeGraphQA:
    def __init__(self):
        """Initialize the QA system with necessary components."""
        print("Initializing KnowledgeGraphQA system...")
        
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo-preview",
            temperature=0
        )
        print("LLM initialized")
        
        self.embeddings = OpenAIEmbeddings()
        print("Embeddings model initialized")
        
        # Initialize vector store
        self.vector_store = SupabaseVectorStore(
            client=supabase,
            embedding=self.embeddings,
            table_name="nodes",
            query_name="match_documents"
        )
        print("Vector store initialized")
        
        # Set up the QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
        print("QA chain initialized")
        
    def load_knowledge_graph(self) -> Dict[str, Any]:
        """Load the entire knowledge graph from Supabase."""
        try:
            print("Loading knowledge graph from Supabase...")
            response = supabase.table("nodes").select("*").execute()
            nodes = {}
            
            for row in response.data:
                nodes[row['id']] = {
                    'title': row['title'],
                    'content': row['content'],
                    'node_type': row['node_type'],
                    'relationships': json.loads(row['relationships']),
                    'metadata': json.loads(row['metadata'])
                }
            
            print(f"Loaded {len(nodes)} nodes from knowledge graph")
            return nodes
        except Exception as e:
            print(f"Error loading knowledge graph: {str(e)}")
            return {}

    def find_related_nodes(self, query: str, nodes: Dict[str, Any]) -> List[str]:
        """Find nodes related to the query using vector similarity and graph relationships."""
        print(f"\nFinding nodes related to query: {query}")
        
        # Get similar nodes using vector search
        print("Performing vector similarity search...")
        similar_docs = self.vector_store.similarity_search(query, k=3)
        print(f"Found {len(similar_docs)} similar documents")
        
        related_node_ids = set()
        
        # Add similar nodes
        for doc in similar_docs:
            print(f"\nProcessing similar document: {doc}")
            if 'id' in doc.metadata:
                node_id = doc.metadata['id']
                related_node_ids.add(node_id)
                print(f"Added node {node_id} from vector search")
                
                # Add directly connected nodes through relationships
                if node_id in nodes:
                    node = nodes[node_id]
                    for rel_type, rel_list in node['relationships'].items():
                        for rel in rel_list:
                            if isinstance(rel, dict):
                                related_node_ids.add(rel['node_id'])
                                print(f"Added related node {rel['node_id']} through relationship {rel_type}")
                            else:
                                related_node_ids.add(rel)
                                print(f"Added related node {rel} through relationship {rel_type}")
        
        print(f"\nTotal related nodes found: {len(related_node_ids)}")
        return list(related_node_ids)

    def build_context(self, related_nodes: List[str], nodes: Dict[str, Any]) -> str:
        """Build context from related nodes for the QA system."""
        print("\nBuilding context from related nodes...")
        context_parts = []
        
        for node_id in related_nodes:
            if node_id in nodes:
                node = nodes[node_id]
                context_parts.append(f"Node {node_id} ({node['node_type']}):")
                context_parts.append(f"Title: {node['title']}")
                context_parts.append(f"Content: {node['content']}")
                context_parts.append("Relationships:")
                
                for rel_type, rel_list in node['relationships'].items():
                    rel_ids = [rel['node_id'] if isinstance(rel, dict) else rel for rel in rel_list]
                    context_parts.append(f"- {rel_type}: {', '.join(rel_ids)}")
                
                context_parts.append("")
        
        context = "\n".join(context_parts)
        print(f"Built context ({len(context)} characters)")
        return context

    def answer_question(self, query: str, chat_history: List = None) -> Dict[str, Any]:
        """Answer a question using the knowledge graph."""
        try:
            print(f"\nProcessing question: {query}")
            
            # Load the full knowledge graph
            nodes = self.load_knowledge_graph()
            
            # Find related nodes
            related_nodes = self.find_related_nodes(query, nodes)
            print(f"Found {len(related_nodes)} related nodes")
            
            # Build context from related nodes
            context = self.build_context(related_nodes, nodes)
            
            # Initialize chat history if None
            if chat_history is None:
                chat_history = []
            
            print("Getting answer from QA chain...")
            # Get answer using the QA chain
            result = self.qa_chain({
                "question": query,
                "chat_history": chat_history
            })
            
            print("Answer generated successfully")
            return {
                "answer": result["answer"],
                "sources": [doc.metadata for doc in result.get("source_documents", [])],
                "related_nodes": related_nodes
            }
            
        except Exception as e:
            print(f"Error answering question: {str(e)}")
            return {
                "error": str(e),
                "answer": "죄송합니다. 질문에 답변하는 중 오류가 발생했습니다.",
                "related_nodes": []
            }

# Example usage
if __name__ == "__main__":
    qa_system = KnowledgeGraphQA()
    
    # Example question
    question = "영희와 철수의 관계는 어떻게 되나요?"
    result = qa_system.answer_question(question)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {result['answer']}")
    print("\nRelated Nodes:", result['related_nodes']) 