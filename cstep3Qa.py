import json
import boto3
import numpy as np
import chromadb
import logging
from langchain.embeddings import BedrockEmbeddings
# Removed incorrect imports for unsupported retrievers

# Initialize Bedrock Embeddings and ChromaDB
bedrock_embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="model_knowledge_base")

# Configure logging
logging.basicConfig(filename="qa_system.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

def get_bedrock_embeddings(texts):
    """
    Uses Bedrock Embeddings API to generate embeddings for input texts.
    """
    try:
        embeddings = bedrock_embeddings.embed_documents(texts)
        if not embeddings:
            raise ValueError("Embedding API returned None")
        return embeddings
    except Exception as e:
        logging.error("Error generating embeddings: %s", str(e))
        return None

def load_knowledge_base(kb_file):
    """
    Loads the structured knowledge base JSON file and stores embeddings in ChromaDB.
    """
    try:
        with open(kb_file, "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)
        
        existing_ids = set(collection.get()['ids'])  # Retrieve existing document IDs from ChromaDB
        
        for idx, entry in enumerate(knowledge_base):
            doc_id = str(idx)
            if doc_id in existing_ids:
                continue  # Skip duplicate IDs
            
            text = entry.get("insights", "")
            embedding = get_bedrock_embeddings([json.dumps(text)])
            if embedding:
                collection.add(ids=[doc_id], embeddings=[embedding[0]], metadatas=[{"insights": text}])
        
        logging.info("Knowledge base loaded successfully.")
        return knowledge_base
    except Exception as e:
        logging.error("Error loading knowledge base: %s", str(e))
        return []

def vector_search_retriever(query, top_k=5, score_threshold=0.3):
    """
    Uses ChromaDB for standard vector search to retrieve the most relevant insights.
    Ensures embedding retrieval is successful before proceeding.
    """
    try:
        query_embedding = get_bedrock_embeddings([query])
        if query_embedding is None or not query_embedding:
            logging.error("Query embedding generation failed, skipping vector search.")
            return []
        
        results = collection.query(query_embeddings=[query_embedding[0]], n_results=top_k)
        # Apply a score threshold to filter out low-relevance chunks
        filtered_results = []
        for idx, metadata in enumerate(results.get('metadatas', [])):
            score_list = results.get('distances', [[1.0]])
            score = score_list[idx] if isinstance(score_list, list) and len(score_list) > idx else 1.0 if 'distances' in results and isinstance(results.get('distances', []), list) and len(results.get('distances', [])) > 0 else 1.0  # ChromaDB uses distance (lower is better)
            if score < score_threshold:
                filtered_results.append(metadata.get("insights", "No insights found"))
        
        return filtered_results[:top_k] if filtered_results else []
        
    except Exception as e:
        logging.error("Error retrieving from vector search: %s", str(e))
        return []

def bm25_search_retriever(query, knowledge_base, top_k=5, use_stopwords=True):
    """
    Uses BM25 for keyword-based search to retrieve the most relevant insights.
    """
    try:
        from rank_bm25 import BM25Okapi
        corpus = [entry.get("insights", "") for entry in knowledge_base]
        stop_words = {"the", "is", "in", "and", "to", "of", "a", "for", "on", "with", "as", "by", "an", "at", "from", "or", "but", "not", "be", "are", "this", "that", "it", "we", "you", "he", "she", "they", "them", "can", "will", "just", "so", "if", "than", "because", "about", "while", "during", "before", "after", "above", "below", "under", "over", "again", "further", "then", "once", "here", "there", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"}
          # Fallback if stopwords are not downloaded
        tokenized_corpus = [[word for word in doc.split() if word.lower() not in stop_words] for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        top_results = bm25.get_top_n(query.split(), corpus, n=top_k)
        return top_results
    except Exception as e:
        logging.error("Error retrieving from BM25 search: %s", str(e))
        return []

def search_knowledge_base(query, knowledge_base, top_k=5, vector_score_threshold=0.3):
    """
    Uses an ensemble retriever combining both vector search and BM25 keyword search.
    Uses a weighted scoring mechanism instead of set() to prioritize relevance.
    """
    vector_results = vector_search_retriever(query, top_k, vector_score_threshold)
    bm25_results = bm25_search_retriever(query, knowledge_base, top_k)
    
    combined_results = {}
    
    # Assign weighted scores: 0.7 weight for vector search, 0.3 weight for BM25
    for idx, doc in enumerate(vector_results):
        vector_weight = 0.7 if vector_results else 0.5
        combined_results[doc] = combined_results.get(doc, 0) + (vector_weight * (top_k - idx))
        vector_weight = 0.7 if vector_results else 0.5
    for idx, doc in enumerate(bm25_results):
        bm25_weight = 0.3 if bm25_results else 0.5
        combined_results[doc] = combined_results.get(doc, 0) + (bm25_weight * (top_k - idx))
    combined_results[doc] = combined_results.get(doc, 0) + (vector_weight * (top_k - idx))
    for idx, doc in enumerate(bm25_results):
        combined_results[doc] = combined_results.get(doc, 0) + (bm25_weight * (top_k - idx))
    
    # Sort results by weighted scores
    sorted_results = sorted(combined_results.keys(), key=lambda x: combined_results[x], reverse=True)
    
    filtered_results = [doc for doc in sorted_results if doc != "No insights found"][:top_k]
    return filtered_results if filtered_results else ["No relevant information found."]

def interactive_qa_system(kb_file):
    """
    Runs an interactive Q&A session where the user can ask questions about the model.
    """
    knowledge_base = load_knowledge_base(kb_file)
    
    print("\nInteractive Q&A System: Type 'exit' to quit.")
    while True:
        query = input("Ask a question: ")
        if query.lower() == 'exit':
            break
        
        results = search_knowledge_base(query, knowledge_base)
        
        print("\nTop Retrieved Insights:\n")
        for idx, result in enumerate(results, 1):
            print(f"{idx}. {result}")
        print("-" * 80)

# Example usage
kb_file = "extracted_text_knowledge_base.json"
interactive_qa_system(kb_file)
