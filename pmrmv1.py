# Improved Code: Processing Large Whitepapers with AWS Bedrock Claude 3 Haiku

After reviewing the previous solution for processing 300+ page whitepapers, I've identified several areas for improvement. Here's an enhanced version with better error handling, more reliable section extraction, and improved JSON parsing from Claude responses:

```python
import boto3
import PyPDF2
import json
import re
import time
import os
from datetime import datetime

def extract_text_from_large_pdf(pdf_path, chunk_size=25, output_dir=None):
    """
    Extracts text from a large PDF file using PyPDF2, processing in chunks.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Number of pages to process at once
        output_dir: If provided, saves individual chunk texts to files for inspection
    
    Returns:
        Dictionary with total pages and text chunks
    """
    try:
        print(f"Opening PDF: {pdf_path}")
        with open(pdf_path, 'rb') as file:
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                print(f"Successfully opened PDF with {total_pages} pages")
            except Exception as e:
                print(f"Error opening PDF with PyPDF2: {e}")
                return None
            
            chunks = []
            # Process in chunks to manage memory
            for start_page in range(0, total_pages, chunk_size):
                end_page = min(start_page + chunk_size, total_pages)
                print(f"Processing pages {start_page + 1} to {end_page} ({end_page - start_page} pages)")
                
                chunk_text = ""
                for page_num in range(start_page, end_page):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        # Verify we got some text
                        if page_text and len(page_text.strip()) > 0:
                            chunk_text += page_text + "\n\n"
                        else:
                            print(f"Warning: Page {page_num + 1} yielded no text. It may be a scanned image or have text in a format PyPDF2 can't extract.")
                    except Exception as e:
                        print(f"Error extracting text from page {page_num + 1}: {e}")
                
                # Basic quality check
                if len(chunk_text.strip())  1:
                    print(f"Warning: Extracted very little text ({len(chunk_text)} chars) from {end_page - start_page} pages. PDF might be scanned or contains images.")
                
                # Save chunk for inspection if requested
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    chunk_file = f"{output_dir}/chunk_{start_page + 1}_to_{end_page}.txt"
                    with open(chunk_file, 'w', encoding='utf-8') as f:
                        f.write(chunk_text)
                    print(f"Saved chunk text to {chunk_file}")
                
                chunks.append({
                    "pages": f"{start_page + 1}-{end_page}",
                    "text": chunk_text
                })
                
                # Small delay to prevent memory issues
                time.sleep(0.1)
            
            return {
                "total_pages": total_pages,
                "chunks": chunks
            }
    except Exception as e:
        print(f"Error in extract_text_from_large_pdf: {e}")
        return None

def identify_section_names(pdf_text_chunks):
    """
    Analyzes the document to identify actual section names.
    
    This helps handle different naming conventions across whitepapers.
    """
    # Common section name patterns
    section_patterns = [
        r'\n\s*\d+\.\d*\s+([A-Z][A-Za-z\s]+)',  # Numbered sections (e.g., "1.2 Application Outputs")
        r'\n([A-Z][A-Z\s]{2,})\s*\n',            # ALL CAPS section headers
        r'\n([A-Z][A-Za-z\s]+:)\s*\n'            # Title with colon (e.g., "Application Outputs:")
    ]
    
    # Search first 3 chunks for section naming pattern
    search_text = ""
    for chunk in pdf_text_chunks["chunks"][:3]:
        search_text += chunk["text"]
    
    detected_sections = []
    for pattern in section_patterns:
        matches = re.finditer(pattern, search_text, re.MULTILINE)
        for match in matches:
            section_name = match.group(1).strip()
            if len(section_name) > 3 and section_name not in detected_sections:
                detected_sections.append(section_name)
    
    print(f"Detected possible section names: {detected_sections}")
    return detected_sections

def extract_section_from_pdf(pdf_text_chunks, target_section_name):
    """
    Extracts a specific section by searching for section header patterns.
    Handles multiple naming variations for better success rate.
    """
    section_text = ""
    section_found = False
    started_at_chunk = -1
    
    # Generate variations of the section name to search for
    section_variations = [
        target_section_name,
        target_section_name.upper(),
        target_section_name.lower(),
        target_section_name.title(),
        f"{target_section_name}:",
        f"{target_section_name.upper()}:"
    ]
    
    # Look for numbered patterns like "4.2 Application Outputs"
    for i in range(1, 13):  # Try section numbers 1-12
        for j in range(0, 10):  # Subsections 0-9
            section_variations.append(f"{i}.{j} {target_section_name}")
            section_variations.append(f"{i}.{j}. {target_section_name}")
            section_variations.append(f"{i} {target_section_name}")
            section_variations.append(f"{i}. {target_section_name}")
    
    # Create regex pattern using all variations
    pattern_parts = [re.escape(var) for var in section_variations]
    section_pattern = re.compile(f"({'|'.join(pattern_parts)})[^\n]*\n", re.IGNORECASE)
    
    # Find the next section pattern - looking for either numbered sections or all caps headers
    next_section_pattern = re.compile(r'\n\s*\d+\.\d*\s+[A-Z][^\n]*\n|\n[A-Z][A-Z\s]{2,}\n')
    
    for i, chunk in enumerate(pdf_text_chunks["chunks"]):
        chunk_text = chunk["text"]
        
        # If we already found the section in a previous chunk, continue collecting
        if section_found:
            # Look for the start of the next section
            next_section_match = next_section_pattern.search(chunk_text)
            if next_section_match:
                # Add text up to the next section
                section_text += chunk_text[:next_section_match.start()]
                print(f"Found end of section '{target_section_name}' in chunk {i+1} (pages {chunk['pages']})")
                break
            else:
                # Add whole chunk if next section not found
                section_text += chunk_text
        else:
            # Look for the section start
            section_start_match = section_pattern.search(chunk_text)
            if section_start_match:
                section_found = True
                started_at_chunk = i
                section_start_pos = section_start_match.start()
                print(f"Found section '{target_section_name}' in chunk {i+1} (pages {chunk['pages']})")
                
                # Look for next section in same chunk
                next_section_match = next_section_pattern.search(chunk_text[section_start_pos:])
                if next_section_match:
                    section_text = chunk_text[section_start_pos:section_start_pos + next_section_match.start()]
                    print(f"Found complete section '{target_section_name}' in chunk {i+1}")
                    break
                else:
                    section_text = chunk_text[section_start_pos:]
    
    # Safety check - if we've collected from more than 10 chunks, we probably missed the end marker
    if section_found and started_at_chunk >= 0 and (len(pdf_text_chunks["chunks"]) - started_at_chunk) > 10:
        print(f"Warning: Section '{target_section_name}' spans many chunks (10+). May have missed the end marker.")
        # Limit to reasonable size (10 chunks from the start)
        if started_at_chunk + 10  max_text_tokens:
        truncation_chars = int(max_text_tokens * 4)
        print(f"Warning: {section_name} section is very long ({len(section_text)} chars, ~{int(estimated_tokens)} tokens)")
        print(f"Truncating to {truncation_chars} chars (~{max_text_tokens} tokens)")
        section_text = section_text[:truncation_chars]
    
    # Custom prompts based on section type
    if "output" in section_name.lower():
        prompt = f"""
        You are an expert auditor reviewing a model whitepaper. I need you to extract all model outputs from the "{section_name}" section.
        
        Focus on identifying:
        1. The name of each output
        2. A description of each output
        3. Any details about how the output is calculated or used
        
        Return ONLY a JSON object with this structure:
        {{
            "model_outputs": [
                {{"name": "output_name", "description": "output description"}}
            ]
        }}
        
        Place your JSON response between triple backticks with the json tag. Example:
        ```
        {{
            "model_outputs": []
        }}
        ```
        
        Here is the {section_name} section text:
        
        {section_text}
        """
    elif "assumption" in section_name.lower():
        prompt = f"""
        You are an expert auditor reviewing a model whitepaper. I need you to extract all assumptions from the "{section_name}" section.
        
        List all assumptions made during model development. Include both explicit assumptions and any implicit assumptions you can infer.
        
        Return ONLY a JSON object with this structure:
        {{
            "assumptions": [
                "assumption_1",
                "assumption_2"
            ]
        }}
        
        Place your JSON response between triple backticks with the json tag. Example:
        ```
        {{
            "assumptions": []
        }}
        ```
        
        Here is the {section_name} section text:
        
        {section_text}
        """
    elif "limitation" in section_name.lower():
        prompt = f"""
        You are an expert auditor reviewing a model whitepaper. I need you to extract all limitations from the "{section_name}" section.
        
        Identify any constraints, limitations, or known issues mentioned about the model.
        
        Return ONLY a JSON object with this structure:
        {{
            "limitations": [
                "limitation_1",
                "limitation_2"
            ]
        }}
        
        Place your JSON response between triple backticks with the json tag. Example:
        ```
        {{
            "limitations": []
        }}
        ```
        
        Here is the {section_name} section text:
        
        {section_text}
        """
    elif "input" in section_name.lower():
        prompt = f"""
        You are an expert auditor reviewing a model whitepaper. I need you to extract all model inputs from the "{section_name}" section.
        
        Focus on identifying:
        1. The name of each input
        2. A description of each input
        3. The source or format of the input if mentioned
        
        Return ONLY a JSON object with this structure:
        {{
            "model_inputs": [
                {{"name": "input_name", "description": "input description", "source": "source of input (if mentioned)"}}
            ]
        }}
        
        Place your JSON response between triple backticks with the json tag. Example:
        ```
        {{
            "model_inputs": []
        }}
        ```
        
        Here is the {section_name} section text:
        
        {section_text}
        """
    else:
        prompt = f"""
        You are an expert auditor reviewing a model whitepaper. I need you to extract key information from the "{section_name}" section.
        
        Identify and summarize the most important points. Focus on technical details, methodologies, or any critical information relevant to model implementation and testing.
        
        Return ONLY a JSON object with this structure:
        {{
            "key_points": [
                "point_1",
                "point_2"
            ]
        }}
        
        Place your JSON response between triple backticks with the json tag. Example:
        ```
        {{
            "key_points": []
        }}
        ```
        
        Here is the {section_name} section text:
        
        {section_text}
        """
    
    print(f"Sending {section_name} to Claude for analysis...")
    response = invoke_claude(bedrock_client, prompt)
    
    if not response:
        print(f"Failed to get a response from Claude for {section_name}")
        return {"error": f"No response from Claude for {section_name}"}
    
    # Extract and parse JSON from Claude's response
    extracted_data = extract_json_from_response(response)
    
    if not extracted_data:
        print(f"Failed to extract JSON from Claude's response for {section_name}")
        # Save the raw response for debugging
        debug_file = f"debug_{section_name.replace(' ', '_').lower()}_response.txt"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(response)
        print(f"Saved raw Claude response to {debug_file}")
        return {"error": f"Could not parse JSON from Claude response", "raw_response_saved": debug_file}
    
    print(f"Successfully extracted information from {section_name}")
    return extracted_data

def process_large_whitepaper(pdf_path, output_dir="./output"):
    """
    Main function to process a large whitepaper
    """
    # Create timestamp for outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== Starting whitepaper processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Output will be saved to {output_dir}")
    
    # Step 1: Extract text from PDF
    print("\n=== Extracting text from whitepaper ===")
    pdf_text_chunks = extract_text_from_large_pdf(pdf_path, chunk_size=25, output_dir=f"{output_dir}/chunks")
    
    if not pdf_text_chunks:
        print("Failed to extract text from whitepaper. Exiting.")
        return
    
    # Step 2: Get Bedrock client
    bedrock_client = get_bedrock_client()
    if not bedrock_client:
        print("Failed to initialize Bedrock client. Exiting.")
        return
    
    # Step 3: Identify section naming patterns in the document
    detected_sections = identify_section_names(pdf_text_chunks)
    
    # Step 4: Define target sections to extract
    # Map standard section names to potential variations found in the document
    target_sections = [
        "Application Outputs",  # For reconciliation testing, this is the most critical
        "Model Application Overview",
        "Assumptions",
        "Limitations",
        "Introduction",
        "Executive Summary",
        "Model Input"  # Try to find input section
    ]
    
    # Match detected sections against our targets
    section_matches = {}
    for target in target_sections:
        target_lower = target.lower()
        for detected in detected_sections:
            detected_lower = detected.lower()
            if (target_lower in detected_lower or
                detected_lower in target_lower or
                any(word in detected_lower for word in target_lower.split())):
                section_matches[target] = detected
                break
    
    print(f"\n=== Matched sections to extract ===")
    for target, match in section_matches.items():
        print(f"Target: '{target}' -> Found: '{match}'")
    
    # Step 5: Extract and analyze each section
    model_information = {}
    
    print("\n=== Extracting and analyzing sections ===")
    for section_name in target_sections:
        print(f"\nProcessing section: {section_name}")
        
        # Use matched section name if available, otherwise try the target name
        search_name = section_matches.get(section_name, section_name)
        
        # Extract section text
        section_text = extract_section_from_pdf(pdf_text_chunks, search_name)
        
        if section_text:
            # Save raw section text
            section_filename = f"{output_dir}/{section_name.replace(' ', '_').lower()}.txt"
            with open(section_filename, 'w', encoding='utf-8') as f:
                f.write(section_text)
            print(f"Saved raw section text to {section_filename}")
            
            # Extract information from section using Claude
            section_info = extract_model_info_from_section(section_text, section_name, bedrock_client)
            model_information[section_name.replace(' ', '_').lower()] = section_info
        else:
            print(f"No content found for section '{section_name}'")
            model_information[section_name.replace(' ', '_').lower()] = {"error": "Section not found in document"}
    
    # Step 6: Save model information
    model_info_file = f"{output_dir}/model_information.json"
    with open(model_info_file, 'w', encoding='utf-8') as f:
        json.dump(model_information, f, indent=4)
    
    print(f"\n=== Model information extracted and saved to {model_info_file} ===")
    print(f"Processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 7: Generate summary of findings
    summary = {
        "whitepaper_filename": os.path.basename(pdf_path),
        "processing_timestamp": timestamp,
        "total_pages": pdf_text_chunks["total_pages"],
        "sections_processed": len(model_information),
        "sections_found": [k for k, v in model_information.items() if "error" not in v],
        "sections_missing": [k for k, v in model_information.items() if "error" in v],
        "key_outputs_count": len(model_information.get("application_outputs", {}).get("model_outputs", [])),
        "assumptions_count": len(model_information.get("assumptions", {}).get("assumptions", [])),
        "limitations_count": len(model_information.get("limitations", {}).get("limitations", []))
    }
    
    summary_file = f"{output_dir}/processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Processing summary saved to {summary_file}")
    return model_information

# Example usage in a SageMaker notebook
if __name__ == "__main__":
    # Local path to the whitepaper PDF file
    whitepaper_path = "./data/model_whitepaper.pdf"
    output_dir = "./output"
    
    # Process the whitepaper
    model_info = process_large_whitepaper(whitepaper_path, output_dir)
```

## Key Improvements for 300+ Page Documents

1. **Better Section Detection**
   - Now automatically identifies section naming patterns in the document
   - Handles multiple variations of section names (e.g., "6.2 Application Outputs", "APPLICATION OUTPUTS")
   - Falls back to keyword search if section headers aren't found

2. **Enhanced PDF Processing**
   - Quality checks for extracted text to detect scanned pages or images
   - Options to save chunk files for manual inspection
   - Better error reporting for problematic pages

3. **Optimized Claude Integration**
   - Improved prompt engineering with explicit JSON formatting instructions
   - Robust JSON extraction from various response formats
   - Rate limiting handling with exponential backoff

4. **Memory & Token Management**
   - Proper estimation of tokens (not just characters)
   - Respects Claude 3 Haiku's ~200K token context window
   - Better chunk management for very large sections

5. **Comprehensive Error Handling**
   - Detailed error messages at each processing stage
   - Debug file output for troubleshooting
   - Summary report of successes and failures

6. **Progress Tracking & Documentation**
   - Detailed logging throughout the process
   - Timestamps for processing steps
   - Summary statistics for the extraction results

## How to Use in SageMaker

1. Upload your whitepaper PDF to your SageMaker environment
2. Install required packages:
   ```
   !pip install PyPDF2 boto3
   ```
3. Copy the code to a notebook cell and run
4. Check the output directory for:
   - Extracted section text files
   - JSON file with structured model information
   - Processing summary
   - Chunk files (for debugging if needed)

## Expected Outputs

1. **model_information.json**: Structured information extracted from the whitepaper
2. **processing_summary.json**: Overview of the extraction process
3. **[section_name].txt**: Raw text for each extracted section
4. **chunks/**: Directory with individual chunk text files (if enabled)

This improved version handles large whitepapers more reliably and provides detailed feedback throughout the process, making it easier to troubleshoot issues with specific documents.

---
Answer from Perplexity: pplx.ai/share
