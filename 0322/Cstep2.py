import json
import os
import logging
from typing import List, Tuple
from uuid import uuid4
import concurrent.futures

import boto3
import faiss
import numpy as np

# Configuration
EMBED_MODEL_ID = "amazon.titan-embed-text-v1"
CHUNK_SIZE = 300          # max characters per chunk
BATCH_SIZE = 10           # embeddings per Titan request (1 at a time)
MAX_WORKERS = 8           # threads for parallelism

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === STEP 1: Multithreaded chunk extraction ===
def process_block_subset(blocks: List[dict]) -> List[Tuple[str, str]]:
    chunk_subset = []
    for block in blocks:
        if block["BlockType"] in {"LINE", "CELL"}:
            text = block.get("Text", "").strip()
            if text:
                chunk_id = str(uuid4())
                chunk_subset.append((chunk_id, text[:CHUNK_SIZE]))
    return chunk_subset

def extract_text_chunks_from_textract(textract_path: str) -> List[Tuple[str, str]]:
    with open(textract_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    blocks = data.get("Blocks", [])
    chunk_size = len(blocks) // MAX_WORKERS
    block_splits = [blocks[i:i+chunk_size] for i in range(0, len(blocks), chunk_size)]

    all_chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_block_subset, blk) for blk in block_splits]
        for future in concurrent.futures.as_completed(futures):
            all_chunks.extend(future.result())

    logger.info(f"✅ Extracted {len(all_chunks)} chunks from Textract (threaded).")
    return all_chunks

# === STEP 2: Multithreaded embedding via Bedrock ===
def embed_text(text: str) -> List[float]:
    bedrock = boto3.client("bedrock-runtime")
    try:
        body = {"inputText": text}
        response = bedrock.invoke_model(
            modelId=EMBED_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        result = json.loads(response["body"].read())
        return result["embedding"]
    except Exception as e:
        logger.warning(f"Embedding failed: {text[:60]}... — {e}")
        return [0.0] * 1536

def get_embeddings_from_bedrock(texts: List[str], max_workers: int = MAX_WORKERS) -> List[List[float]]:
    embeddings = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(embed_text, text) for text in texts]
        for future in concurrent.futures.as_completed(futures):
            embeddings.append(future.result())
    return embeddings

# === STEP 3: FAISS index builder ===
def build_faiss_index(chunks: List[Tuple[str, str]], output_dir: str = "faiss_index"):
    os.makedirs(output_dir, exist_ok=True)
    chunk_ids = [c[0] for c in chunks]
    texts = [c[1] for c in chunks]
    embeddings = get_embeddings_from_bedrock(texts)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    # Save index and mapping files
    faiss.write_index(index, os.path.join(output_dir, "index.faiss"))
    with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(dict(zip(chunk_ids, texts)), f, indent=2)
    with open(os.path.join(output_dir, "id_map.json"), "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, indent=2)

    logger.info(f"✅ FAISS index built and saved with {len(texts)} vectors to '{output_dir}'")

# === Run the pipeline ===
if __name__ == "__main__":
    textract_path = "textract_output.json"
    chunks = extract_text_chunks_from_textract(textract_path)
    build_faiss_index(chunks)








import json
import faiss
import numpy as np
import boto3
from typing import List

# Configuration
EMBED_MODEL_ID = "amazon.titan-embed-text-v1"
FAISS_INDEX_PATH = "faiss_index/index.faiss"
CHUNKS_PATH = "faiss_index/chunks.json"
ID_MAP_PATH = "faiss_index/id_map.json"
TOP_K = 5

# === Load FAISS index ===
def load_faiss_index(index_path: str):
    return faiss.read_index(index_path)

# === Load metadata (chunk IDs and texts) ===
def load_chunks(chunk_file: str, id_map_file: str):
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunk_dict = json.load(f)
    with open(id_map_file, "r", encoding="utf-8") as f:
        id_map = json.load(f)
    return chunk_dict, id_map

# === Embed user query via Bedrock Titan ===
def embed_query(query: str) -> List[float]:
    bedrock = boto3.client("bedrock-runtime")
    body = {"inputText": query}
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    return json.loads(response["body"].read())["embedding"]

# === Search FAISS index ===
def search_index(query: str,
                 index_path: str = FAISS_INDEX_PATH,
                 chunk_file: str = CHUNKS_PATH,
                 id_map_file: str = ID_MAP_PATH,
                 top_k: int = TOP_K) -> List[str]:

    index = load_faiss_index(index_path)
    chunk_dict, id_map = load_chunks(chunk_file, id_map_file)
    vector = embed_query(query)

    D, I = index.search(np.array([vector]).astype("float32"), top_k)

    results = []
    for idx in I[0]:
        if idx < len(id_map):
            chunk_id = id_map[idx]
            text = chunk_dict.get(chunk_id, "")
            results.append(text)

    return results

# === Run from CLI ===
if __name__ == "__main__":
    user_query = input("Ask a question about the document: ")
    top_chunks = search_index(user_query)

    print("\nTop matching document chunks:\n" + "-"*40)
    for i, chunk in enumerate(top_chunks):
        print(f"\nChunk {i+1}:\n{chunk}")







import json
from semantic_search import search_index
import boto3

# Claude Model ID
CLAUDE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# === Build Claude Prompt ===
def build_claude_prompt(question: str, context_chunks: list) -> str:
    prompt = (
        "You are a document analysis assistant.\n"
        "Based on the extracted content below, answer the user’s question.\n\n"
        "Context:\n"
    )

    for i, chunk in enumerate(context_chunks):
        prompt += f"Chunk {i+1}:\n{chunk}\n\n"

    prompt += f"User Question: {question}\n\n"
    prompt += "Answer clearly and concisely. If the answer is not found, reply 'Not found.'"
    return prompt

# === Claude Bedrock Call ===
def query_claude(prompt: str) -> str:
    bedrock = boto3.client("bedrock-runtime")

    response = bedrock.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.2
        })
    )

    body = json.loads(response['body'].read())
    return body['content'][0]['text']

# === Main Q&A Function ===
def ask_question(question: str, top_k: int = 5):
    print(f"🔍 Searching document for: {question}")
    chunks = search_index(question, top_k=top_k)

    print("🧠 Sending context to Claude...")
    prompt = build_claude_prompt(question, chunks)
    answer = query_claude(prompt)

    print("\n=== Claude's Answer ===")
    print(answer)

# === Run as CLI ===
if __name__ == "__main__":
    print("Claude Document Q&A Assistant")
    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() in ["exit", "quit"]:
            break
        ask_question(q)





