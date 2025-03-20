Phase 2: LLM-Based Whitepaper Analysis and Key Highlights Extraction
In this next phase, we will implement a system where an LLM analyzes the whitepaper, extracts key highlights, and provides a summary of what the document contains. This will help users understand the core content and structure of the whitepaper without reading the entire document.

Implementation Approach
Goals
Extract Key Highlights:

Identify important sections, key findings, and critical insights from the whitepaper.

Summarize methodologies, results, and conclusions.

Document Structure Analysis:

Provide an overview of what the document contains (e.g., sections, tables, figures).

Automated Insights:

Derive actionable insights, such as validation methodologies or assumptions.

Testing Plan:

Ensure accuracy, completeness, and relevance of extracted highlights.

System Architecture
Step 1: Preprocessing
Split the whitepaper into manageable chunks using chunking logic (similar to Phase 1).

Extract metadata (e.g., titles, headers) to identify document structure.

Step 2: LLM Analysis
Use an LLM (e.g., Claude or GPT) to analyze each chunk.

Generate structured outputs for:

Key highlights

Section summaries

Document structure overview

Step 3: Consolidation
Combine chunk-level insights into a single comprehensive summary.

Organize results into structured categories (e.g., Introduction, Methodology, Results).

Step 4: Output Generation
Generate a final report with:

Key highlights

Document structure overview

Actionable insights

Implementation Code
Here’s how we can implement this system:


##################################################
import boto3
import json
import logging
import re
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_nltk_dependencies():
    """Ensure NLTK dependencies are downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    Uses a simple heuristic: ~4 characters per token for English text.
    """
    return len(text) // 4 + 1

def token_based_chunking(text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks based on token count rather than character count.
    """
    download_nltk_dependencies()
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_token_count(sentence)
        
        # If adding this sentence would exceed chunk size and we already have content,
        # finish current chunk and start a new one
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Include overlap from the end of the previous chunk
            overlap_tokens = 0
            overlap_sentences = []
            
            for s in reversed(current_chunk):
                s_tokens = estimate_token_count(s)
                if overlap_tokens + s_tokens <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens
                else:
                    break
            
            current_chunk = overlap_sentences
            current_tokens = overlap_tokens
        
        # Add the sentence to the current chunk
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def smart_chunk_text(text, chunk_size=4000, overlap=200, strategy="token"):
    """
    Enhanced chunking with multiple strategies.
    
    Args:
        text (str): Text to chunk
        chunk_size (int): Maximum size (chars or tokens) per chunk
        overlap (int): Overlap between chunks
        strategy (str): Chunking strategy: "token", "paragraph", or "sentence"
        
    Returns:
        List[str]: List of text chunks
    """
    if strategy == "token":
        return token_based_chunking(text, chunk_size, overlap)
    
    elif strategy == "sentence":
        download_nltk_dependencies()
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Calculate overlap
                overlap_size = 0
                overlap_sentences = []
                
                for s in reversed(current_chunk):
                    if overlap_size + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    elif strategy == "paragraph":
        # Split text by paragraphs (or double newline)
        paragraphs = text.split("\n\n")
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:  # Skip empty paragraphs
                continue
                
            paragraph_size = len(paragraph)
            
            # Handle case where a single paragraph is too large
            if paragraph_size > chunk_size and not current_chunk:
                # Recursively chunk this large paragraph using sentence strategy
                sub_chunks = smart_chunk_text(paragraph, chunk_size, overlap, "sentence")
                chunks.extend(sub_chunks)
                continue
                
            # If adding this paragraph would exceed chunk size and we already have content,
            # finish current chunk and start a new one
            if current_size + paragraph_size > chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                
                # Include some overlap from previous chunk
                overlap_size = 0
                overlap_paragraphs = []
                
                # Start from the end and work backwards to add paragraphs for overlap
                for p in reversed(current_chunk):
                    if overlap_size + len(p) <= overlap:
                        overlap_paragraphs.insert(0, p)
                        overlap_size += len(p)
                    else:
                        break
                        
                current_chunk = overlap_paragraphs
                current_size = overlap_size
            
            # Add the paragraph to the current chunk
            current_chunk.append(paragraph)
            current_size += paragraph_size
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
    
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Use 'token', 'paragraph', or 'sentence'.")

def analyze_chunk_with_retry(chunk, retry_count=0, max_retries=2, reduce_size_factor=0.75):
    """
    Analyze a chunk with auto-retry and size reduction logic to handle "input too long" errors.
    """
    try:
        return analyze_chunk_for_highlights(chunk)
    except Exception as e:
        error_str = str(e).lower()
        if ("input too long" in error_str or "input is too long" in error_str) and retry_count < max_retries:
            # Reduce chunk size and try again
            reduced_size = int(len(chunk) * reduce_size_factor)
            logging.warning(f"Input too long error. Reducing chunk size from {len(chunk)} to {reduced_size} characters.")
            
            # Split the chunk into smaller pieces
            smaller_chunks = smart_chunk_text(chunk, chunk_size=reduced_size, overlap=200, strategy="sentence")
            
            if len(smaller_chunks) == 1 and len(smaller_chunks[0]) >= len(chunk) * 0.95:
                # If we couldn't meaningfully reduce the size, try a more aggressive approach
                sentences = nltk.sent_tokenize(chunk)
                middle_idx = len(sentences) // 2
                first_half = " ".join(sentences[:middle_idx])
                
                logging.warning(f"Attempting with half the chunk: {len(first_half)} characters")
                return analyze_chunk_with_retry(first_half, retry_count + 1, max_retries, reduce_size_factor)
            
            # Process the first smaller chunk
            return analyze_chunk_with_retry(smaller_chunks[0], retry_count + 1, max_retries, reduce_size_factor)
        else:
            logging.error(f"Error processing chunk: {str(e)}")
            return {
                "highlights": [],
                "summary": [f"Error processing chunk: {str(e)}"],
                "methodologies": [],
                "assumptions": [],
                "figures_tables": []
            }


def analyze_chunk_for_highlights(chunk):
    """
    Analyzes a single chunk of text using Claude via AWS Bedrock to extract key highlights.
    """
    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    prompt = f"""
    You are an expert tasked with analyzing a section of a financial model whitepaper. 
    The goal is to extract key highlights, summarize the content, and identify technical details.

    Read the following content carefully:

    {chunk}

    Tasks:
    1. Extract **key highlights** from this section. These should be concise sentences capturing the most important points.
    2. Summarize the **main points** in bullet format. Ensure the summary captures actionable insights.
    3. Identify any **methodologies** mentioned in this section.
    4. Extract any **assumptions** explicitly stated or implied in this section.
    5. If figures or tables are referenced, describe their purpose briefly.

    Format your response as JSON with these keys:
    {{
      "highlights": [<key_highlight_1>, <key_highlight_2>, ...],
      "summary": [<bullet_point_1>, <bullet_point_2>, ...],
      "methodologies": [<methodology_1>, <methodology_2>, ...],
      "assumptions": [<assumption_1>, <assumption_2>, ...],
      "figures_tables": [<figure_or_table_reference>]
    }}

    Return only valid JSON output.
    """
    
    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 3000,
                "temperature": 0.3  # Lower temperature for more consistent JSON formatting
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        
        # Extract content from response
        content = response_body["content"]
        
        # Handle JSON content which might be wrapped in markdown code blocks
        json_pattern = r'``````'
        json_match = re.search(json_pattern, content, re.DOTALL)
        
        if json_match:
            # Extract JSON from code block
            json_str = json_match.group(1).strip()
        else:
            # If not in code block, use the whole content
            json_str = content.strip()
        
        # Parse the JSON
        result = json.loads(json_str)
        
        # Validate structure
        for key in ["highlights", "summary", "methodologies", "assumptions", "figures_tables"]:
            if key not in result:
                result[key] = []
        
        return result
    
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        raise e

def process_chunks_in_parallel(chunks, max_workers=3):
    """
    Process multiple chunks in parallel with error handling.
    """
    results = []
    processed_count = 0
    failure_count = 0
    
    logging.info(f"Processing {len(chunks)} chunks with {max_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chunk = {executor.submit(analyze_chunk_with_retry, chunk): i 
                         for i, chunk in enumerate(chunks)}
        
        # Process results as they complete
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            processed_count += 1
            
            try:
                result = future.result()
                results.append(result)
                
                # Log progress
                if processed_count % 5 == 0 or processed_count == len(chunks):
                    logging.info(f"Processed {processed_count}/{len(chunks)} chunks")
                
            except Exception as e:
                failure_count += 1
                logging.error(f"Chunk {chunk_idx} processing failed with error: {str(e)}")
                # Add empty result with error information
                results.append({
                    "highlights": [],
                    "summary": [f"Failed to process chunk {chunk_idx}: {str(e)}"],
                    "methodologies": [],
                    "assumptions": [],
                    "figures_tables": []
                })
    
    logging.info(f"Finished processing all chunks ({failure_count} failures out of {len(chunks)} chunks)")
    
    return results

def main():
    """
    Main workflow for analyzing a whitepaper and extracting key highlights.
    """
    # Load extracted text from file
    extracted_text_path = "./extracted_content/extracted_text.txt"
    
    try:
        with open(extracted_text_path, "r", encoding="utf-8") as f:
            extracted_text = f.read()
        
        logging.info(f"Loaded extracted text from {extracted_text_path}")
        
        # Step 1: Chunk the text using token-based strategy (better for Claude models)
        logging.info("Chunking extracted text...")
        chunks = smart_chunk_text(
            extracted_text, 
            chunk_size=4000,  # Conservative chunk size for Claude
            overlap=200,  
            strategy="token"  # Use token-based chunking
        )
        
        logging.info(f"Created {len(chunks)} chunks")
        
        # Fallback if only one chunk was created
        if len(chunks) == 1:
            logging.warning("Only one chunk created. Attempting alternative chunking strategy...")
            # Try sentence-based chunking as a fallback
            chunks = smart_chunk_text(extracted_text, chunk_size=4000, overlap=200, strategy="sentence")
            logging.info(f"Sentence-based chunking created {len(chunks)} chunks")
            
            # Force splitting if still only one chunk
            if len(chunks) == 1:
                logging.warning("Still only one chunk. Forcing split into smaller pieces...")
                sentences = nltk.sent_tokenize(extracted_text)
                n_sentences = len(sentences)
                
                # Split into multiple chunks of approximately equal size
                max_sentences_per_chunk = min(50, max(5, n_sentences // 3))
                new_chunks = []
                
                for i in range(0, n_sentences, max_sentences_per_chunk):
                    chunk = " ".join(sentences[i:i+max_sentences_per_chunk])
                    new_chunks.append(chunk)
                
                chunks = new_chunks
                logging.info(f"Forced chunking created {len(chunks)} chunks")
        
        # Step 2: Analyze chunks in parallel with retry logic
        logging.info("Analyzing chunks in parallel...")
        results = process_chunks_in_parallel(chunks, max_workers=3)
        
        # Step 3: Consolidate results with deduplication
        logging.info("Consolidating results...")
        consolidated_results = consolidate_highlights(results)
        
        # Step 4: Export results to Word document
        output_file_path = "./whitepaper_highlights.docx"
        logging.info(f"Exporting highlights to Word document at {output_file_path}...")
        export_to_word_highlights(consolidated_results, output_file_path)
        
        print(f"Highlights exported successfully to {output_file_path}")
    
    except Exception as e:
        logging.error(f"Error during whitepaper analysis workflow: {str(e)}")

if __name__ == "__main__":
   main()

