import boto3
import PyPDF2
import json
import re
import time
import os

def extract_text_from_large_pdf(pdf_path, chunk_size=25):
    """
    Extracts text from a large PDF file using PyPDF2, processing in chunks.
    chunk_size: Number of pages to process at once
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            print(f"PDF has {total_pages} pages")
            
            chunks = []
            # Process in chunks to manage memory
            for start_page in range(0, total_pages, chunk_size):
                end_page = min(start_page + chunk_size, total_pages)
                print(f"Processing pages {start_page + 1} to {end_page}")
                
                chunk_text = ""
                for page_num in range(start_page, end_page):
                    page = pdf_reader.pages[page_num]
                    chunk_text += page.extract_text() + "\n\n"
                
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
        print(f"Error extracting text from PDF: {e}")
        return None

def extract_toc_from_pdf(pdf_text_chunks):
    """
    Attempts to extract table of contents from the first few chunks.
    """
    # Look at first 2 chunks (usually where TOC is located)
    toc_search_text = ""
    for chunk in pdf_text_chunks["chunks"][:2]:
        toc_search_text += chunk["text"]
    
    # Various patterns that might indicate TOC
    toc_patterns = [
        r"(?:TABLE OF CONTENTS|Contents|CONTENTS).*?(?=\d+\.\s+\w+)",
        r"(?:TABLE OF CONTENTS|Contents|CONTENTS)(.*?)(?=\n\s*\d+\.\s+\w+)",
        r"(?:TABLE OF CONTENTS|Contents|CONTENTS)(.*?)(?=\n\s*\d+\.0\s+\w+)"
    ]
    
    for pattern in toc_patterns:
        matches = re.search(pattern, toc_search_text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches.group(0)
    
    return "Table of Contents not found"

def extract_section_from_pdf(pdf_text_chunks, section_name):
    """
    Extracts a specific section by searching for section header patterns.
    """
    section_text = ""
    section_found = False
    
    # Create patterns to find the section
    section_start_pattern = re.compile(f'(?:{section_name}|{section_name.upper()})[^\n]*\n', re.IGNORECASE)
    next_section_pattern = re.compile(r'\n\s*\d+\.\d*\s+[A-Z][^\n]*\n')
    
    for i, chunk in enumerate(pdf_text_chunks["chunks"]):
        chunk_text = chunk["text"]
        
        # If we already found the section in a previous chunk, look for next section
        if section_found:
            # Look for the start of the next section
            next_section_match = next_section_pattern.search(chunk_text)
            if next_section_match:
                # Add text up to the next section
                section_text += chunk_text[:next_section_match.start()]
                break
            else:
                # Add whole chunk if next section not found
                section_text += chunk_text
        else:
            # Look for the section start
            section_start_match = section_start_pattern.search(chunk_text)
            if section_start_match:
                section_found = True
                section_start_pos = section_start_match.start()
                
                # Look for next section in same chunk
                next_section_match = next_section_pattern.search(chunk_text[section_start_pos:])
                if next_section_match:
                    section_text = chunk_text[section_start_pos:section_start_pos + next_section_match.start()]
                    break
                else:
                    section_text = chunk_text[section_start_pos:]
    
    if not section_found:
        print(f"Section '{section_name}' not found in the document")
        return None
    
    return section_text

def get_bedrock_client():
    """
    Creates and returns a Bedrock client for invoking Claude 3 Haiku model.
    """
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'  # Change to your region
    )
    return bedrock_client

def invoke_claude(bedrock_client, prompt, model_id="anthropic.claude-3-haiku-20240307-v1:0", max_tokens=4096):
    """
    Invokes Claude 3 Haiku model via AWS Bedrock.
    """
    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        
        response_body = json.loads(response["body"].read().decode("utf-8"))
        return response_body["content"][0]["text"]
    except Exception as e:
        print(f"Error invoking Claude model: {e}")
        return None

def extract_model_info_from_section(section_text, section_name, bedrock_client):
    """
    Extracts information from a specific section using Claude.
    """
    if not section_text:
        return {"error": f"No text found for section {section_name}"}
    
    # Truncate if too long (Claude has context limits)
    if len(section_text) > 100000:
        print(f"Warning: {section_name} section is very long ({len(section_text)} chars). Truncating to 100,000 chars.")
        section_text = section_text[:100000]
    
    # Custom prompts based on section type
    if section_name.lower() == "application outputs":
        prompt = f"""
        You are an expert auditor reviewing a model whitepaper. I need you to extract all model outputs from the "Application Outputs" section.
        
        Focus on identifying:
        1. The name of each output
        2. A description of each output
        3. Any details about how the output is calculated or used
        
        Return your response in JSON format with the following structure:
        {{
            "model_outputs": [
                {{"name": "output_name", "description": "output description"}}
            ]
        }}
        
        Here is the Application Outputs section:
        
        {section_text}
        """
    elif section_name.lower() == "assumptions":
        prompt = f"""
        You are an expert auditor reviewing a model whitepaper. I need you to extract all assumptions from the "Assumptions" section.
        
        List all assumptions made during model development. Include both explicit assumptions and any implicit assumptions you can infer.
        
        Return your response in JSON format with the following structure:
        {{
            "assumptions": [
                "assumption_1",
                "assumption_2",
                ...
            ]
        }}
        
        Here is the Assumptions section:
        
        {section_text}
        """
    elif section_name.lower() == "limitations":
        prompt = f"""
        You are an expert auditor reviewing a model whitepaper. I need you to extract all limitations from the "Limitations" section.
        
        Identify any constraints, limitations, or known issues mentioned about the model.
        
        Return your response in JSON format with the following structure:
        {{
            "limitations": [
                "limitation_1",
                "limitation_2",
                ...
            ]
        }}
        
        Here is the Limitations section:
        
        {section_text}
        """
    else:
        prompt = f"""
        You are an expert auditor reviewing a model whitepaper. I need you to extract key information from the "{section_name}" section.
        
        Identify and summarize the most important points from this section. Focus on technical details, methodologies, or any critical information.
        
        Return your response in JSON format with the following structure:
        {{
            "key_points": [
                "point_1",
                "point_2",
                ...
            ]
        }}
        
        Here is the {section_name} section:
        
        {section_text}
        """
    
    response = invoke_claude(bedrock_client, prompt)
    
    # Parse JSON from Claude's response
    try:
        # Look for JSON in the response
        json_match = re.search(r'``````', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no JSON code block, try to extract JSON directly
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
        
        # Parse the JSON
        extracted_data = json.loads(json_str)
        return extracted_data
    except json.JSONDecodeError:
        print(f"Error parsing JSON from Claude response for {section_name}")
        return {"error": f"Failed to parse Claude response for {section_name}"}

def process_large_whitepaper(pdf_path, output_dir="./output"):
    """
    Main function to process a large whitepaper
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract text from PDF
    print("Extracting text from whitepaper...")
    pdf_text_chunks = extract_text_from_large_pdf(pdf_path)
    
    if not pdf_text_chunks:
        print("Failed to extract text from whitepaper.")
        return
    
    # Step 2: Extract TOC to identify key sections
    print("Extracting Table of Contents...")
    toc = extract_toc_from_pdf(pdf_text_chunks)
    
    with open(f"{output_dir}/toc.txt", 'w') as f:
        f.write(toc)
    
    # Step 3: Get Bedrock client
    bedrock_client = get_bedrock_client()
    
    # Step 4: Extract key sections and analyze with Claude
    key_sections = [
        "Executive Summary",
        "Introduction",
        "Model Application Overview", 
        "Application Outputs",
        "Assumptions",
        "Limitations"
    ]
    
    model_information = {}
    
    for section_name in key_sections:
        print(f"Processing section: {section_name}")
        
        # Extract section text
        section_text = extract_section_from_pdf(pdf_text_chunks, section_name)
        
        if section_text:
            # Save raw section text
            with open(f"{output_dir}/{section_name.replace(' ', '_').lower()}.txt", 'w') as f:
                f.write(section_text)
            
            # Extract information from section using Claude
            section_info = extract_model_info_from_section(section_text, section_name, bedrock_client)
            model_information[section_name.replace(' ', '_').lower()] = section_info
    
    # Step 5: Save model information
    with open(f"{output_dir}/model_information.json", 'w') as f:
        json.dump(model_information, f, indent=4)
    
    print(f"Model information extracted and saved to {output_dir}/model_information.json")
    return model_information

# Example usage
if __name__ == "__main__":
    # Local path to the whitepaper PDF file
    whitepaper_path = "./data/model_whitepaper.pdf"
    output_dir = "./output"
    
    # Process the whitepaper
    model_info = process_large_whitepaper(whitepaper_path, output_dir)
