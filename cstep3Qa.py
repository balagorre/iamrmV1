
import json
import boto3
import numpy as np
import chromadb
from langchain.embeddings import BedrockEmbeddings

# Initialize Bedrock Embeddings and ChromaDB
bedrock_embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="model_knowledge_base")

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
        
        return knowledge_base
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return []

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
        return f"Error generating response: {e}"

def interactive_qa_system(kb_file):
    """
    Runs an interactive Q&A session where the user can ask questions about the model.
    Saves chat history for context-aware follow-up questions.
    """
    load_knowledge_base(kb_file)
    
    chat_history = []
    print("\nModel Q&A System: Type 'exit' to quit.")
    
    while True:
        query = input("Ask a question: ")
        if query.lower() == 'exit':
            break
        
        # Include chat history in the search for context-aware responses
        full_context = "\n".join([f"Q: {ch['question']}\nA: {ch['answer']}" for ch in chat_history])
        context_query = f"Previous Context:\n{full_context}\n\nCurrent Query: {query}" if chat_history else query
        
        relevant_chunks = search_knowledge_base(context_query)
        response = generate_response_with_claude3(query, relevant_chunks)
        
        chat_history.append({"question": query, "answer": response})
        
        print("\nAnswer:\n", response)
        print("-" * 80)
    
    # Save chat history for future reference
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(chat_history, f, indent=4)
        
    print("Chat history saved to chat_history.json.")

# Example usage
kb_file = "extracted_text_knowledge_base.json"
interactive_qa_system(kb_file)










import json
import boto3
import numpy as np
import chromadb
from chromadb.utils import embedding_functions

# Initialize Amazon Titan Embeddings and ChromaDB
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="model_knowledge_base")

def get_titan_embeddings(texts):
    """
    Uses Amazon Titan Embeddings via Bedrock to generate embeddings for the input texts.
    """
    payload = {"inputText": texts}
    response = bedrock.invoke_model(
        body=json.dumps(payload),
        modelId='amazon.titan-embed-text-v1',
        accept='application/json',
        contentType='application/json'
    )
    response_body = json.loads(response['body'].read().decode('utf-8'))
    return np.array(response_body['embedding'])

def load_knowledge_base(kb_file):
    """
    Loads the structured knowledge base JSON file and stores embeddings in ChromaDB.
    """
    try:
        with open(kb_file, "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)
        
        for idx, entry in enumerate(knowledge_base):
            embedding = get_titan_embeddings([json.dumps(entry["insights"])])[0]
            collection.add(ids=[str(idx)], embeddings=[embedding.tolist()], metadatas=[{"insights": entry["insights"]}])
        
        return knowledge_base
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return []

def search_knowledge_base(query, top_k=5):
    """
    Uses ChromaDB to retrieve the most relevant insights using Titan embeddings.
    """
    query_embedding = get_titan_embeddings([query])[0]
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
    
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
        return f"Error generating response: {e}"

def interactive_qa_system(kb_file):
    """
    Runs an interactive Q&A session where the user can ask questions about the model.
    Saves chat history for context-aware follow-up questions.
    """
    load_knowledge_base(kb_file)
    
    chat_history = []
    print("\nModel Q&A System: Type 'exit' to quit.")
    
    while True:
        query = input("Ask a question: ")
        if query.lower() == 'exit':
            break
        
        # Include chat history in the search for context-aware responses
        full_context = "\n".join([f"Q: {ch['question']}\nA: {ch['answer']}" for ch in chat_history])
        context_query = f"Previous Context:\n{full_context}\n\nCurrent Query: {query}" if chat_history else query
        
        relevant_chunks = search_knowledge_base(context_query)
        response = generate_response_with_claude3(query, relevant_chunks)
        
        chat_history.append({"question": query, "answer": response})
        
        print("\nAnswer:\n", response)
        print("-" * 80)
    
    # Save chat history for future reference
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(chat_history, f, indent=4)
        
    print("Chat history saved to chat_history.json.")

# Example usage
kb_file = "extracted_text_knowledge_base.json"
interactive_qa_system(kb_file)
