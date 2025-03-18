
def search_knowledge_base(query, top_k=5):
    """
    Uses ChromaDB to retrieve the most relevant insights using Bedrock embeddings.
    """
    query_embedding = get_bedrock_embeddings([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    if results["documents"]:
        return [metadata.get("insights", "No insights found") for metadata in results["metadatas"]]
    return ["No relevant information found."]



import json
import boto3
import numpy as np
import chromadb
import logging
from langchain.embeddings import BedrockEmbeddings

# Initialize Bedrock Embeddings and ChromaDB
bedrock_embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="model_knowledge_base")

# Configure logging
logging.basicConfig(filename="qa_system.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_bedrock_embeddings(texts):
    """
    Uses Bedrock Embeddings API to generate embeddings for input texts.
    """
    return bedrock_embeddings.embed_documents(texts)

def load_knowledge_base(kb_file):
    """
    Loads the structured knowledge base JSON file and stores embeddings in ChromaDB.
    """
    try:
        with open(kb_file, "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)
        
        for idx, entry in enumerate(knowledge_base):
            embedding = get_bedrock_embeddings([json.dumps(entry["insights"])])[0]
            collection.add(ids=[str(idx)], embeddings=[embedding], metadatas=[{"insights": entry["insights"]}])
        
        logging.info("Knowledge base loaded successfully with %d entries.", len(knowledge_base))
        return knowledge_base
    except Exception as e:
        logging.error("Error loading knowledge base: %s", str(e))
        return []

def multi_step_retrieval(query, top_k=5):
    """
    Implements multi-step retrieval where the query is decomposed into sub-queries for better reasoning.
    """
    logging.info("User query: %s", query)
    initial_results = search_knowledge_base(query, top_k)
    refined_query = f"Based on these retrieved insights: {json.dumps(initial_results, indent=4)}, generate more specific sub-queries."
    
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    payload = {"prompt": refined_query, "max_tokens": 500, "temperature": 0.3}
    
    try:
        response = bedrock.invoke_model(
            body=json.dumps(payload),
            modelId='anthropic.claude-3-sonnet-2024-02-29',
            accept='application/json',
            contentType='application/json'
        )
        response_body = json.loads(response['body'].read().decode('utf-8'))
        sub_queries = response_body.get('completion', 'No sub-queries generated').split("\n")
        logging.info("Generated sub-queries: %s", sub_queries)
        
        additional_results = []
        for sub_query in sub_queries:
            additional_results.extend(search_knowledge_base(sub_query.strip(), top_k))
        
        return list(set(initial_results + additional_results))
    except Exception as e:
        logging.error("Error generating sub-queries: %s", str(e))
        return initial_results

def search_knowledge_base(query, top_k=5):
    """
    Uses ChromaDB to retrieve the most relevant insights using Bedrock embeddings.
    """
    query_embedding = get_bedrock_embeddings([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    if results["documents"]:
        return [doc["insights"] for doc in results["metadatas"]]
    return ["No relevant information found."]

def generate_response_with_claude3(query, relevant_chunks, max_chunks=3):
    """
    Uses Claude 3 to refine the response based on retrieved insights.
    Limits the number of relevant chunks sent to prevent exceeding token limits and ensure concise responses.
    """
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    
    limited_chunks = relevant_chunks[:max_chunks]  # Limit insights to avoid token overflow
    
    prompt = f"""
    You are an AI assistant helping answer questions about a model whitepaper. Below are relevant model insights:
    
    {json.dumps(limited_chunks, indent=4)}
    
    Based on the above insights, answer the following question:
    {query}
    """
    
    payload = {
        "prompt": prompt,
        "max_tokens": 1000,
        "temperature": 0.5
    }
    
    try:
        response = bedrock.invoke_model(
            body=json.dumps(payload),
            modelId='anthropic.claude-3-sonnet-2024-02-29',
            accept='application/json',
            contentType='application/json'
        )
        response_body = json.loads(response['body'].read().decode('utf-8'))
        return response_body.get('completion', 'No response received')
    except Exception as e:
        logging.error("Error generating response: %s", str(e))
        return f"Error generating response: {e}"

def interactive_qa_system(kb_file):
    """
    Runs an interactive Q&A session where the user can ask questions about the model.
    Saves chat history for context-aware follow-up questions and logs interactions.
    """
    load_knowledge_base(kb_file)
    
    chat_history = []
    print("\nModel Q&A System: Type 'exit' to quit.")
    
    while True:
        query = input("Ask a question: ")
        if query.lower() == 'exit':
            break
        
        # Multi-step retrieval for better context
        relevant_chunks = multi_step_retrieval(query)
        response = generate_response_with_claude3(query, relevant_chunks)
        
        chat_history.append({"question": query, "answer": response})
        logging.info("User query: %s | Response: %s", query, response)
        
        print("\nAnswer:\n", response)
        print("-" * 80)
    
    # Save chat history for future reference
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(chat_history, f, indent=4)
        
    print("Chat history saved to chat_history.json.")
    logging.info("Chat history saved successfully.")

# Example usage
kb_file = "extracted_text_knowledge_base.json"
interactive_qa_system(kb_file)
