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
    high-value information from a financial model whitepaper section. Your goal is to perform a 
    comprehensive extraction of key information, technical details, and structural elements.

    ===== CONTENT TO ANALYZE =====
    {chunk}
    =============================

    ## ANALYSIS STEPS

    ### STEP 1: DEEP CONTENT UNDERSTANDING
    First, carefully read and understand the content. Identify the main topics, key technical concepts, 
    and the overall purpose of this section within a financial model whitepaper.

    ### STEP 2: KEY HIGHLIGHTS EXTRACTION
    Extract 3-7 key highlights that represent the most important information. These should be:
    - Concise yet complete sentences that capture significant findings, conclusions, or insights
    - Focused on quantitative results, key methodologies, or critical assumptions when present
    - Ordered by importance (most significant first)
    - Include exact numerical values and technical terminology as presented in the text

    ### STEP 3: COMPREHENSIVE SUMMARY CREATION
    Create a bullet-point summary that:
    - Covers all major points in logical order
    - Prioritizes actionable insights and decision-relevant information
    - Preserves technical precision while being accessible to financial professionals
    - Avoids redundancy while ensuring completeness

    ### STEP 4: TECHNICAL METHODOLOGY IDENTIFICATION
    Identify all methodologies mentioned:
    - Extract names of algorithms, statistical methods, computational techniques, etc.
    - For each methodology, provide a brief description of how it's used in this context
    - Note implementation details when available (parameters, configurations, etc.)
    - Include references to any benchmarks or validation techniques mentioned

    ### STEP 5: ASSUMPTION EXTRACTION AND CLASSIFICATION
    Carefully identify both explicit and implicit assumptions:
    - Data assumptions (e.g., data quality, completeness, distribution properties)
    - Model assumptions (e.g., statistical properties, feature relationships)
    - Business/domain assumptions (e.g., market conditions, regulatory environment)
    - Implementation assumptions (e.g., computational requirements, system constraints)
    - For each assumption, note if it's explicit (clearly stated) or implicit (reasonably inferred)

    ### STEP 6: FIGURE/TABLE ANALYSIS
    For any figures or tables referenced:
    - Identify the figure/table number and title
    - Describe its purpose and key insights it conveys
    - Note any important data sources or methodologies used to generate it
    - Extract key values or trends represented

    ### STEP 7: CONTEXTUAL RELATIONSHIP MAPPING
    Identify how this section relates to:
    - Previous or subsequent sections (if referenced)
    - Overall model architecture or methodology
    - Validation or testing procedures
    - Implementation considerations

    ### STEP 8: TECHNICAL TERMINOLOGY GLOSSARY
    Extract specialized technical terms with their definitions or contexts:
    - Financial terms
    - Statistical/mathematical concepts
    - Domain-specific terminology
    - Acronyms and abbreviations with their expansions

    ## RESPONSE FORMAT GUIDELINES
    Format your response as a comprehensive JSON object with these keys:

    {
      "highlights": [
        {"text": "<highlight_1>", "importance": "high|medium|low"},
        {"text": "<highlight_2>", "importance": "high|medium|low"},
        ...
      ],
      "summary": [
        "<bullet_point_1>",
        "<bullet_point_2>",
        ...
      ],
      "methodologies": [
        {
          "name": "<methodology_name>",
          "description": "<brief_description>",
          "implementation_details": "<specific_parameters_or_configurations>",
          "validation_approach": "<how_validated_if_mentioned>"
        },
        ...
      ],
      "assumptions": [
        {
          "type": "data|model|business|implementation",
          "assumption": "<assumption_text>",
          "explicit": true|false,
          "impact": "<potential_impact_if_mentioned>",
          "validation": "<validation_approach_if_mentioned>"
        },
        ...
      ],
      "figures_tables": [
        {
          "identifier": "<figure_or_table_number>",
          "title": "<title_if_available>",
          "description": "<what_it_shows>",
          "key_insights": "<important_takeaways>",
          "data_source": "<source_of_data_if_mentioned>"
        },
        ...
      ],
      "context_relationships": [
        {
          "related_to": "<section_or_concept>",
          "relationship_type": "prerequisite|builds_on|supports|contrasts_with|validates",
          "description": "<brief_description_of_relationship>"
        },
        ...
      ],
      "technical_terms": [
        {
          "term": "<technical_term>",
          "category": "financial|statistical|domain-specific|acronym",
          "definition": "<definition_or_context>"
        },
        ...
      ],
      "section_quality": {
        "completeness": 1-5,
        "technical_depth": 1-5,
        "clarity": 1-5,
        "actionability": 1-5
      }
    }

    ## IMPORTANT INSTRUCTIONS
    - Be extremely precise and avoid vague statements
    - Preserve ALL numerical values EXACTLY as stated - never round or simplify
    - Always provide full sentences for highlights and bullet points
    - If information for a specific field is not found, use an empty array [] for list fields or "Not found in context" for text fields
    - Ensure your response is valid JSON format
    - Rate section quality objectively based on content comprehensiveness, technical detail level, clarity of explanation, and actionable information provided

    Return ONLY the JSON output, nothing else.
    """
    
    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000,  # Increased token limit for more detailed output
                "temperature": 0.3   # Lower temperature for more precision
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        return json.loads(response_body["content"])
    
    except Exception as e:
        logging.error(f"Error processing chunk: {str(e)}")
        return {
            "error": f"Failed to process chunk due to error: {str(e)}",
            "chunk_preview": chunk[:200] + "..." if len(chunk) > 200 else chunk  # Include preview for debugging
        }





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
