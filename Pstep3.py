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




import concurrent.futures
import logging
import time

def main():
    """
    Main workflow for analyzing a whitepaper and extracting key highlights with parallel processing.
    
    Returns:
        None
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load extracted text from file
    extracted_text_path = "./extracted_content/extracted_text.txt"
    
    try:
        with open(extracted_text_path, "r", encoding="utf-8") as f:
            extracted_text = f.read()
        
        logging.info(f"Loaded extracted text from {extracted_text_path}")
        
        # Step 1: Chunk the text
        logging.info("Chunking extracted text...")
        chunks = chunk_text(extracted_text)
        logging.info(f"Created {len(chunks)} chunks")
        
        # Step 2: Analyze each chunk using LLM in parallel
        logging.info("Analyzing chunks in parallel...")
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel processing
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks and map them to their chunk indices
            future_to_chunk = {executor.submit(analyze_chunk_for_highlights, chunk): i 
                             for i, chunk in enumerate(chunks)}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append(result)
                    logging.info(f"Processed chunk {chunk_idx+1}/{len(chunks)}")
                except Exception as e:
                    logging.error(f"Chunk {chunk_idx+1} generated an exception: {str(e)}")
        
        end_time = time.time()
        logging.info(f"All chunks processed in {end_time - start_time:.2f} seconds")
        
        # Step 3: Consolidate results across all chunks
        logging.info("Consolidating results...")
        consolidated_results = consolidate_highlights(results)
        
        # Step 4: Export results to Word document
        output_file_path = "./whitepaper_highlights.docx"
        logging.info(f"Exporting highlights to Word document at {output_file_path}...")
        export_to_word_highlights(consolidated_results, output_file_path)
        
        print(f"Highlights exported successfully to {output_file_path}")
    
    except Exception as e:
        logging.error(f"Error during whitepaper analysis workflow: {str(e)}")



def analyze_chunk_for_highlights(chunk):
    """
    Analyzes a single chunk of text using Claude via AWS Bedrock to extract key highlights.
    
    Args:
        chunk (str): A chunk of text to analyze.

    Returns:
        dict: Extracted highlights and structured analysis.
    """
    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    prompt = f"""
    You are a world-class financial analyst and technical documentation expert tasked with extracting 
    high-value information from a financial model whitepaper section.

    ===== CONTENT TO ANALYZE =====
    {chunk}
    =============================

    ## ANALYSIS STEPS
    [Your detailed analysis steps here...]

    ## RESPONSE FORMAT GUIDELINES
    Format your response as a comprehensive JSON object. Here is the expected structure:

    {{
      "highlights": [
        {{
          "text": "This is an example highlight with important information.",
          "importance": "high"
        }},
        {{
          "text": "This is another example highlight with medium importance.",
          "importance": "medium"
        }}
      ],
      "summary": [
        "Example bullet point summarizing a key concept.",
        "Another bullet point with important information."
      ],
      "methodologies": [
        {{
          "name": "Example Methodology",
          "description": "Brief description of the methodology",
          "implementation_details": "Details about implementation",
          "validation_approach": "How it was validated"
        }}
      ],
      "assumptions": [
        {{
          "type": "data",
          "assumption": "Example assumption about data quality",
          "explicit": true,
          "impact": "Potential impact of this assumption",
          "validation": "How this assumption was validated"
        }}
      ],
      "figures_tables": [
        {{
          "identifier": "Figure 1",
          "title": "Example Figure Title",
          "description": "What this figure shows",
          "key_insights": "Important takeaways from this figure",
          "data_source": "Source of data used in this figure"
        }}
      ],
      "context_relationships": [
        {{
          "related_to": "Another section name",
          "relationship_type": "builds_on",
          "description": "How this section relates to others"
        }}
      ],
      "technical_terms": [
        {{
          "term": "Example technical term",
          "category": "financial",
          "definition": "Definition of this technical term"
        }}
      ],
      "section_quality": {{
        "completeness": 5,
        "technical_depth": 4,
        "clarity": 5,
        "actionability": 4
      }}
    }}

    ## IMPORTANT INSTRUCTIONS
    - For the "importance" field, use only one of these values: "high", "medium", or "low"
    - For the "type" field in assumptions, use only one of these values: "data", "model", "business", or "implementation"
    - For "explicit" field in assumptions, use only true or false (boolean values)
    - For "relationship_type", use only one of these values: "prerequisite", "builds_on", "supports", "contrasts_with", or "validates"
    - For "category" in technical terms, use only one of these values: "financial", "statistical", "domain-specific", or "acronym"
    - For section quality scores, use integers between 1 and 5 inclusive
    
    Return ONLY the JSON output, nothing else.
    """
    
    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000,
                "temperature": 0.3
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        return json.loads(response_body["content"])
    
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in response: {str(e)}")
        # Clean response before parsing
        content = response_body["content"]
        # Attempt basic fixes like removing markdown code blocks
        if content.startswith("```
            content = content.split("```json", 1)[1]
        if content.endswith("```
            content = content.rsplit("```", 1)[0]
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON even after cleanup")
            return {"error": f"Failed to process chunk due to invalid JSON: {str(e)}",
                   "chunk_preview": chunk[:200] + "..." if len(chunk) > 200 else chunk}
    
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        return {"error": f"Failed to process chunk due to error: {str(e)}",
               "chunk_preview": chunk[:200] + "..." if len(chunk) > 200 else chunk}






def chunk_text(text, chunk_size=10000, overlap=500):
    """
    Splits text into manageable chunks for processing by LLM.
    
    Args:
        text (str): Full text extracted from the whitepaper.
        chunk_size (int): Maximum size of each chunk in characters.
        overlap (int): Overlap between chunks to preserve context.

    Returns:
        list: List of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


import boto3
import json
import logging






def consolidate_highlights(results):
    """
    Consolidates highlights from all chunks into a single structured output.
    
    Args:
        results (list): List of results from individual chunks.

    Returns:
        dict: Consolidated highlights and structured analysis.
    """
    consolidated = {
        "highlights": [],
        "summary": [],
        "methodologies": [],
        "assumptions": [],
        "figures_tables": []
    }
    
    for result in results:
        if result.get("error"):
            logging.warning(f"Skipping failed result: {result['error']}")
            continue
        
        consolidated["highlights"].extend(result.get("highlights", []))
        consolidated["summary"].extend(result.get("summary", []))
        consolidated["methodologies"].extend(result.get("methodologies", []))
        consolidated["assumptions"].extend(result.get("assumptions", []))
        consolidated["figures_tables"].extend(result.get("figures_tables", []))
    
    return consolidated

from docx import Document

def export_to_word_highlights(consolidated_results, output_file_path):
    """
    Exports consolidated highlights to a Word document.
    
    Args:
        consolidated_results (dict): Consolidated structured data from all chunks.
        output_file_path (str): Path to save the Word document.

    Returns:
        None
    """
    doc = Document()
    
    # Add title
    doc.add_heading('Whitepaper Highlights Report', level=1)
    
    # Add highlights section
    doc.add_heading('Key Highlights', level=2)
    for highlight in consolidated_results.get("highlights", []):
        doc.add_paragraph(f"- {highlight}")
    
    # Add summary section
    doc.add_heading('Summary', level=2)
    for bullet_point in consolidated_results.get("summary", []):
        doc.add_paragraph(f"- {bullet_point}")
    
    # Add methodologies section
    doc.add_heading('Methodologies', level=2)
    for methodology in consolidated_results.get("methodologies", []):
        doc.add_paragraph(f"- {methodology}")
    
    # Add assumptions section
    doc.add_heading('Assumptions', level=2)
    for assumption in consolidated_results.get("assumptions", []):
        doc.add_paragraph(f"- {assumption}")
    
    # Add figures/tables section
    doc.add_heading('Figures/Tables', level=2)
    for reference in consolidated_results.get("figures_tables", []):
        doc.add_paragraph(f"- {reference}")
    
    # Save the document
    doc.save(output_file_path)


def main():
    """
    Main workflow for analyzing a whitepaper and extracting key highlights.
    
    Returns:
        None
    """
    
    # Load extracted text from file
    extracted_text_path = "./extracted_content/extracted_text.txt"
    
    try:
        with open(extracted_text_path, "r", encoding="utf-8") as f:
            extracted_text = f.read()
        
        logging.info(f"Loaded extracted text from {extracted_text_path}")
        
        # Step 1: Chunk the text
        logging.info("Chunking extracted text...")
        chunks = chunk_text(extracted_text)
        
        # Step 2: Analyze each chunk using LLM
        logging.info("Analyzing chunks...")
        results = [analyze_chunk_for_highlights(chunk) for chunk in chunks]
        
        # Step 3: Consolidate results across all chunks
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
