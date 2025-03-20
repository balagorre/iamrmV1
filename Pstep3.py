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
    You are an expert tasked with analyzing a section of a financial model whitepaper. 
    The goal is to extract key highlights, summarize the content, and identify technical details.

    Read the following content carefully:

    {chunk}

    Tasks:
    1. Extract **key highlights** from this section. These should be concise sentences capturing the most important points (e.g., findings, results, or conclusions).
    2. Summarize the **main points** in bullet format. Ensure the summary captures actionable insights and avoids redundant information.
    3. Identify any **methodologies** mentioned in this section (e.g., algorithms, statistical methods, or computational techniques). Provide their names and brief descriptions.
    4. Extract any **assumptions** explicitly stated or implied in this section (e.g., data assumptions, model assumptions).
    5. If figures or tables are referenced, describe their purpose briefly (e.g., "Table 1 shows performance metrics across test datasets").

    Instructions:
    - Be precise and avoid vague statements.
    - Preserve numerical values exactly as stated (do not round or simplify).
    - Use clear formatting for your response.
    - If specific information is missing in this section, explicitly state "Not found in context."

    Format your response as JSON with these keys:
    {
      "highlights": [<key_highlight_1>, <key_highlight_2>, ...],
      "summary": [<bullet_point_1>, <bullet_point_2>, ...],
      "methodologies": [<methodology_1>, <methodology_2>, ...],
      "assumptions": [<assumption_1>, <assumption_2>, ...],
      "figures_tables": [<figure_or_table_reference>]
    }

    Return only valid JSON output."""
    
    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 3000,
                "temperature": 0.5
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        return json.loads(response_body["content"])
    
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        return {"error": f"Failed to process chunk due to error: {str(e)}"}

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
