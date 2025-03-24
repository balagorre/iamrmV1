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
