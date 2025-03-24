import json
import os
import logging
from typing import List, Tuple
from uuid import uuid4

import boto3
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBED_MODEL_ID = "amazon.titan-embed-text-v1"
CHUNK_SIZE = 300  # Adjust based on LLM token sensitivity

def extract_text_chunks_from_textract(textract_path: str, max_chunk_len: int = CHUNK_SIZE) -> List[Tuple[str, str]]:
    with open(textract_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    blocks = data.get("Blocks", [])
    for block in blocks:
        if block["BlockType"] in {"LINE", "CELL"}:
            text = block.get("Text", "").strip()
            if text:
                chunk_id = str(uuid4())
                chunks.append((chunk_id, text[:max_chunk_len]))
    logger.info(f"✅ Extracted {len(chunks)} chunks from Textract.")
    return chunks

def get_embeddings_from_bedrock(texts: List[str], batch_size: int = 10) -> List[List[float]]:
    bedrock = boto3.client("bedrock-runtime")
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for text in batch:
            body = {"inputText": text}
            try:
                response = bedrock.invoke_model(
                    modelId=EMBED_MODEL_ID,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body)
                )
                result = json.loads(response["body"].read())
                embeddings.append(result["embedding"])
            except Exception as e:
                logger.warning(f"Embedding failed for text: {text[:50]}... — {e}")
                embeddings.append([0.0] * 1536)  # Zero vector fallback
    return embeddings

def build_faiss_index(chunks: List[Tuple[str, str]], output_dir: str = "faiss_index"):
    os.makedirs(output_dir, exist_ok=True)
    chunk_ids = [c[0] for c in chunks]
    texts = [c[1] for c in chunks]
    embeddings = get_embeddings_from_bedrock(texts)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    # Save FAISS index
    faiss.write_index(index, os.path.join(output_dir, "index.faiss"))

    # Save chunk ID → text map
    with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(dict(zip(chunk_ids, texts)), f, indent=2)

    # Save vector ID → chunk ID map
    with open(os.path.join(output_dir, "id_map.json"), "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, indent=2)

    logger.info(f"✅ FAISS index saved with {len(texts)} vectors at '{output_dir}'")

if __name__ == "__main__":
    textract_path = "textract_output.json"  # Path to your saved Textract response
    chunks = extract_text_chunks_from_textract(textract_path)
    build_faiss_index(chunks)
