import boto3

def upload_pdf_to_s3(local_file_path, bucket_name, s3_folder):
    """
    Uploads a local PDF file to an S3 bucket under a specified folder.
    
    Args:
        local_file_path (str): Path to the local file on the SageMaker instance.
        bucket_name (str): Name of the S3 bucket.
        s3_folder (str): Folder path in the S3 bucket where the file will be uploaded.

    Returns:
        str: The full S3 key of the uploaded file if successful, None otherwise.
    """
    s3 = boto3.client('s3')
    s3_key = f"{s3_folder}/{local_file_path.split('/')[-1]}"  # Generate S3 key based on folder and filename
    
    try:
        print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}...")
        s3.upload_file(local_file_path, bucket_name, s3_key)
        print(f"File uploaded successfully to s3://{bucket_name}/{s3_key}.")
        return s3_key
    except Exception as e:
        print(f"Error uploading file to S3: {str(e)}")
        return None

# Example usage
local_file_path = "./mrecon/data/whitepaper.pdf"
bucket_name = "your-s3-bucket-name"
s3_folder = "home/genai009"

s3_key = upload_pdf_to_s3(local_file_path, bucket_name, s3_folder)
if s3_key:
    print(f"File uploaded successfully. S3 Key: {s3_key}")
else:
    print("File upload failed.")





def start_textract_job(bucket_name, s3_key):
    """
    Starts a Textract job for analyzing a document in S3.
    
    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): Key of the file in the S3 bucket.

    Returns:
        str: The Textract Job ID if successful, None otherwise.
    """
    textract = boto3.client('textract')
    
    try:
        print("Starting Textract job...")
        response = textract.start_document_analysis(
            DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': s3_key}},
            FeatureTypes=['TABLES', 'FORMS']  # Extract tables and forms
        )
        job_id = response['JobId']
        print(f"Textract Job ID: {job_id}")
        return job_id
    except Exception as e:
        print(f"Error starting Textract job: {str(e)}")
        return None

# Example usage
if s3_key:
    textract_job_id = start_textract_job(bucket_name, s3_key)





import time

def wait_for_textract_job(job_id):
    """
    Waits for a Textract job to complete and returns its status.
    
    Args:
        job_id (str): The Textract Job ID.

    Returns:
        str: The final status of the Textract job ('SUCCEEDED' or 'FAILED').
    """
    textract = boto3.client('textract')
    
    while True:
        try:
            response = textract.get_document_analysis(JobId=job_id)
            status = response['JobStatus']
            if status in ['SUCCEEDED', 'FAILED']:
                print(f"Textract job completed with status: {status}")
                return status
            print(f"Job status: {status}. Waiting...")
            time.sleep(5)  # Poll every 5 seconds
        except Exception as e:
            print(f"Error checking Textract job status: {str(e)}")
            return None

# Example usage
if textract_job_id:
    job_status = wait_for_textract_job(textract_job_id)





def retrieve_textract_results(job_id):
    """
    Retrieves results from a completed Textract job.
    
    Args:
        job_id (str): The Textract Job ID.

    Returns:
        list: A list of JSON responses containing extracted data.
    """
    textract = boto3.client('textract')
    results = []
    
    try:
        response = textract.get_document_analysis(JobId=job_id)
        results.append(response)
        
        next_token = response.get('NextToken')
        while next_token:
            response = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
            results.append(response)
            next_token = response.get('NextToken')
        
        print("Textract results retrieved successfully.")
        return results
    except Exception as e:
        print(f"Error retrieving Textract results: {str(e)}")
        return None

# Example usage
if textract_job_id and job_status == "SUCCEEDED":
    textract_results = retrieve_textract_results(textract_job_id)







import os

def process_textract_results(results, output_dir):
    """
    Processes Textract results and saves extracted text and tables locally.
    
    Args:
        results (list): List of JSON responses from Textract.
        output_dir (str): Directory where processed content will be saved.
        
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_text = ""
    
    # Process each result block
    for result in results:
        blocks = result['Blocks']
        
        for block in blocks:
            block_type = block['BlockType']
            
            if block_type == 'LINE':  # Extract text lines
                extracted_text += block['Text'] + "\n"
            
            elif block_type == 'TABLE':  # Extract table data
                table_data = []
                cells = [cell for cell in blocks if cell['BlockType'] == 'CELL']
                for cell in cells:
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    text = cell.get('Text', '')
                    table_data.append((row_index, col_index, text))
                
                # Convert table data into a structured format (CSV-like)
                max_row_index = max([cell[0] for cell in table_data])
                max_col_index = max([cell[1] for cell in table_data])
                
                table_matrix = [["" for _ in range(max_col_index)] for _ in range(max_row_index)]
                for row_idx, col_idx, text in table_data:
                    table_matrix[row_idx-1][col_idx-1] = text
                
                # Save table as CSV
                table_file_path = os.path.join(output_dir, f"table_{len(table_matrix)}.csv")
                with open(table_file_path, "w", encoding="utf-8") as f:
                    for row in table_matrix:
                        f.write(",".join(row) + "\n")
    
    # Save extracted text
    text_file_path = os.path.join(output_dir, "extracted_text.txt")
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

# Example usage
output_dir = "./extracted_content"
if textract_results:
    process_textract_results(textract_results, output_dir)


#####
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=10000, overlap=500):
    """
    Splits text into manageable chunks for processing by Claude.
    
    Args:
        text (str): Full text extracted from the whitepaper.
        chunk_size (int): Maximum size of each chunk in tokens.
        overlap (int): Overlap between chunks to preserve context.

    Returns:
        list: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def create_prompt_xml(chunk, chunk_id):
    """
    Creates a detailed prompt for analyzing a chunk of text using Claude.
    
    Args:
        chunk (str): A chunk of text to analyze.
        chunk_id (int): Unique identifier for the chunk.

    Returns:
        str: A formatted prompt string.
    """
    prompt = f"""
    You are an expert auditor reviewing a financial model whitepaper. Analyze the following content:

    {chunk}

    Tasks:
    1. Provide a summary of this content.
    2. Identify all model inputs and describe their purpose and format (e.g., numerical, categorical).
    3. Identify all model outputs and describe their purpose and format (e.g., numerical, categorical).
    4. List all assumptions made in this content and explain their significance.
    5. Highlight any limitations mentioned in this content and explain their impact on the model.

    Instructions:
    - If specific information is missing in this chunk, state "Not found in context."
    - Provide your response in XML format with the following structure:

      <response chunk_id="{chunk_id}">
        <summary>
          <text>This section describes...</text>
        </summary>
        <model_inputs>
          <input>
            <name>input_1</name>
            <description>Description of input_1</description>
            <format>numerical</format>
          </input>
          <input>
            <name>input_2</name>
            <description>Description of input_2</description>
            <format>categorical</format>
          </input>
        </model_inputs>
        <model_outputs>
          <output>
            <name>output_1</name>
            <description>Description of output_1</description>
            <format>numerical</format>
          </output>
        </model_outputs>
        <assumptions>
          <assumption>Assumption_1</assumption>
          <assumption>Assumption_2</assumption>
        </assumptions>
        <limitations>
          <limitation>Limitation_1</limitation>
          <limitation>Limitation_2</limitation>
        </limitations>
      </response>

    Return only valid XML output."""
    
    return prompt

import xml.etree.ElementTree as ET

def parse_xml_response(response_text):
    """
    Parses an XML response from Claude into structured data.
    
    Args:
        response_text (str): The XML response from Claude.

    Returns:
        dict: Parsed structured data including summary, inputs, outputs, assumptions, and limitations.
    """
    try:
        root = ET.fromstring(response_text)  # Parse the XML string
        
        parsed_data = {
            "chunk_id": root.attrib.get("chunk_id", ""),
            "summary": root.find("summary/text").text if root.find("summary/text") is not None else "",
            "model_inputs": [],
            "model_outputs": [],
            "assumptions": [],
            "limitations": []
        }
        
        # Parse model inputs
        for input_elem in root.findall("model_inputs/input"):
            parsed_data["model_inputs"].append({
                "name": input_elem.find("name").text,
                "description": input_elem.find("description").text,
                "format": input_elem.find("format").text
            })
        
        # Parse model outputs
        for output_elem in root.findall("model_outputs/output"):
            parsed_data["model_outputs"].append({
                "name": output_elem.find("name").text,
                "description": output_elem.find("description").text,
                "format": output_elem.find("format").text
            })
        
        # Parse assumptions
        for assumption_elem in root.findall("assumptions/assumption"):
            parsed_data["assumptions"].append(assumption_elem.text)
        
        # Parse limitations
        for limitation_elem in root.findall("limitations/limitation"):
            parsed_data["limitations"].append(limitation_elem.text)
        
        return parsed_data
    
    except Exception as e:
        print(f"Error parsing XML response: {str(e)}")
        return {"error": f"Failed to parse response: {response_text}"}




import boto3
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)

def analyze_chunk_with_claude_xml(chunk, chunk_id):
    """
    Analyzes a single chunk of text using Claude via AWS Bedrock.
    
    Args:
        chunk (str): A chunk of text to analyze.
        chunk_id (int): Unique identifier for the chunk.

    Returns:
        dict: Parsed structured data including summary, inputs/outputs, assumptions, and limitations.
    """
    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    prompt = create_prompt_xml(chunk, chunk_id)
    
    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        
        # Parse XML response
        return parse_xml_response(response_body["content"])
    
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_id}: {str(e)}")
        return {"error": f"Failed to process chunk {chunk_id}"}

def analyze_whitepaper_parallel_xml(extracted_text):
    """
    Processes chunks of extracted text in parallel using multithreading.
    
    Args:
        extracted_text (str): Full text extracted from the whitepaper.

    Returns:
        list: Combined analysis results from all chunks.
    """
    chunks = chunk_text(extracted_text)
    
    results = []
    
    logging.info(f"Processing {len(chunks)} chunks...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(analyze_chunk_with_claude_xml, chunk, i + 1)
            for i, chunk in enumerate(chunks)
        ]
        
        for future in futures:
            result = future.result()
            results.append(result)
    
    logging.info("All chunks processed.")
    
    return results

# Example usage
with open("./extracted_content/extracted_text.txt", "r", encoding="utf-8") as f:
    extracted_text = f.read()

whitepaper_analysis_parallel_xml = analyze_whitepaper_parallel_xml(extracted_text)

# Save analysis results to a file
with open("./extracted_content/whitepaper_analysis_parallel.xml.json", "w", encoding="utf-8") as f:
    json.dump(whitepaper_analysis_parallel_xml, f, indent=2)

print("Whitepaper analysis saved.")


def consolidate_results(results):
    """
    Consolidates results from all chunks into a single structured output.
    
    Args:
        results (list): List of results from individual chunks.

    Returns:
        dict: Consolidated results including summaries, inputs/outputs, assumptions, and limitations.
    """
    consolidated = {
        "summary": [],
        "model_inputs": [],
        "model_outputs": [],
        "assumptions": [],
        "limitations": []
    }
    
    for result in results:
        if result.get("error"):
            logging.warning(f"Skipping failed result: {result['error']}")
            continue
        
        consolidated["summary"].append(result.get("summary", "").strip())
        consolidated["model_inputs"].extend(result.get("model_inputs", []))
        consolidated["model_outputs"].extend(result.get("model_outputs", []))
        consolidated["assumptions"].extend(result.get("assumptions", []))
        consolidated["limitations"].extend(result.get("limitations", []))
    
    return consolidated

# Consolidate results
final_analysis = consolidate_results(whitepaper_analysis_parallel_xml)

# Save consolidated results
with open("./extracted_content/final_whitepaper_analysis.json", "w", encoding="utf-8") as f:
    json.dump(final_analysis, f, indent=2)

print("Final whitepaper analysis saved.")









import boto3
import json
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def analyze_chunk_with_claude(chunk, chunk_id, model_id="anthropic.claude-3-haiku-20240307-v1:0"):
    """
    Analyzes a single chunk of text using Claude via AWS Bedrock.
    
    Args:
        chunk (str): A chunk of text to analyze.
        chunk_id (int): Unique identifier for the chunk.
        model_id (str): The Claude model ID in AWS Bedrock.

    Returns:
        dict: Structured insights including summary, model inputs/outputs, assumptions, and limitations.
    """
    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    prompt = create_prompt(chunk).replace("<chunk_id>", str(chunk_id))
    
    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        
        # Parse JSON response from Claude
        return json.loads(response_body["content"])
    
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_id}: {str(e)}")
        return {"error": f"Failed to process chunk {chunk_id}"}

def analyze_whitepaper_parallel(extracted_text, max_threads=10):
    """
    Analyzes the entire whitepaper by processing chunks in parallel.
    
    Args:
        extracted_text (str): Full text extracted from the whitepaper.
        max_threads (int): Maximum number of threads for parallel processing.

    Returns:
        list: Combined analysis results from all chunks.
    """
    chunks = chunk_text(extracted_text)
    
    results = []
    
    logging.info(f"Processing {len(chunks)} chunks with {max_threads} threads...")
    
    with ThreadPoolExecutor(max_threads) as executor:
        futures = [
            executor.submit(analyze_chunk_with_claude, chunk, i + 1)
            for i, chunk in enumerate(chunks)
        ]
        
        for future in futures:
            result = future.result()
            results.append(result)
    
    logging.info("All chunks processed.")
    
    return results

# Example usage
with open("./extracted_content/extracted_text.txt", "r", encoding="utf-8") as f:
    extracted_text = f.read()

whitepaper_analysis_parallel = analyze_whitepaper_parallel(extracted_text)

# Save analysis results to a file
with open("./extracted_content/whitepaper_analysis_parallel.json", "w", encoding="utf-8") as f:
    json.dump(whitepaper_analysis_parallel, f, indent=2)

print("Whitepaper analysis saved.")





def consolidate_results(results):
    """
    Consolidates results from all chunks into a single structured output.
    
    Args:
        results (list): List of results from individual chunks.

    Returns:
        dict: Consolidated results including summaries, inputs/outputs, assumptions, and limitations.
    """
    consolidated = {
        "summary": [],
        "model_inputs": [],
        "model_outputs": [],
        "assumptions": [],
        "limitations": []
    }
    
    for result in results:
        if result.get("error"):
            logging.warning(f"Skipping failed result: {result['error']}")
            continue
        
        consolidated["summary"].append(result.get("summary", "").strip())
        consolidated["model_inputs"].extend(result.get("model_inputs", []))
        consolidated["model_outputs"].extend(result.get("model_outputs", []))
        consolidated["assumptions"].extend(result.get("assumptions", []))
        consolidated["limitations"].extend(result.get("limitations", []))
    
    return consolidated





















##########################################


import boto3
import time
import json
import os

def upload_pdf_to_s3(pdf_path, bucket_name, s3_key):
    """
    Uploads the PDF to an S3 bucket for processing by Textract.
    """
    s3 = boto3.client('s3')
    try:
        print(f"Uploading {pdf_path} to S3 bucket {bucket_name}...")
        s3.upload_file(pdf_path, bucket_name, s3_key)
        print(f"File uploaded successfully to {bucket_name}/{s3_key}.")
        return True
    except Exception as e:
        print(f"Error uploading file to S3: {str(e)}")
        return False

def start_textract_job(bucket_name, s3_key):
    """
    Starts a Textract job to analyze the PDF for text, tables, and forms.
    """
    textract = boto3.client('textract')
    try:
        print("Starting Textract job...")
        response = textract.start_document_analysis(
            DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': s3_key}},
            FeatureTypes=['TABLES', 'FORMS']  # Extract tables and forms
        )
        job_id = response['JobId']
        print(f"Textract Job ID: {job_id}")
        return job_id
    except Exception as e:
        print(f"Error starting Textract job: {str(e)}")
        return None

def wait_for_textract_job(job_id):
    """
    Waits for the Textract job to complete and returns the status.
    """
    textract = boto3.client('textract')
    while True:
        try:
            response = textract.get_document_analysis(JobId=job_id)
            status = response['JobStatus']
            if status in ['SUCCEEDED', 'FAILED']:
                print(f"Textract job completed with status: {status}")
                return status
            print(f"Job status: {status}. Waiting...")
            time.sleep(5)
        except Exception as e:
            print(f"Error checking Textract job status: {str(e)}")
            return None

def retrieve_textract_results(job_id):
    """
    Retrieves Textract results after the job is completed.
    Handles pagination for large documents.
    """
    textract = boto3.client('textract')
    results = []
    
    try:
        response = textract.get_document_analysis(JobId=job_id)
        results.append(response)
        
        next_token = response.get('NextToken')
        while next_token:
            response = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
            results.append(response)
            next_token = response.get('NextToken')
        
        print("Textract results retrieved successfully.")
        return results
    except Exception as e:
        print(f"Error retrieving Textract results: {str(e)}")
        return None

def process_textract_results(results, output_dir):
    """
    Processes Textract results to extract text, tables, and forms.
    Saves extracted data into separate files for downstream analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize containers for extracted data
    extracted_text = ""
    tables = []
    
    # Process each result block
    for result in results:
        blocks = result['Blocks']
        
        for block in blocks:
            block_type = block['BlockType']
            
            if block_type == 'LINE':  # Extract text lines
                extracted_text += block['Text'] + "\n"
            
            elif block_type == 'TABLE':  # Extract table data
                table_data = []
                cells = [cell for cell in blocks if cell['BlockType'] == 'CELL']
                for cell in cells:
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    text = cell.get('Text', '')
                    table_data.append((row_index, col_index, text))
                
                # Convert table data into a structured format (CSV-like)
                max_row_index = max([cell[0] for cell in table_data])
                max_col_index = max([cell[1] for cell in table_data])
                
                table_matrix = [["" for _ in range(max_col_index)] for _ in range(max_row_index)]
                for row_idx, col_idx, text in table_data:
                    table_matrix[row_idx-1][col_idx-1] = text
                
                tables.append(table_matrix)
    
    # Save extracted text
    with open(os.path.join(output_dir, "extracted_text.txt"), "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    # Save extracted tables as CSV files
    for i, table in enumerate(tables):
        table_path = os.path.join(output_dir, f"table_{i+1}.csv")
        with open(table_path, "w", encoding="utf-8") as f:
            for row in table:
                f.write(",".join(row) + "\n")
    
    print("Extracted content saved successfully.")
    
# Main function to execute the enhanced step 1 workflow
def extract_content_from_pdf(pdf_path, bucket_name, s3_key, output_dir):
    """
    Executes the full workflow of uploading a PDF to S3,
    processing it with Textract, and saving extracted content locally.
    """
    # Step 1: Upload the PDF to S3
    if not upload_pdf_to_s3(pdf_path, bucket_name, s3_key):
        print("Failed to upload PDF to S3. Exiting.")
        return
    
    # Step 2: Start Textract job
    job_id = start_textract_job(bucket_name, s3_key)
    if not job_id:
        print("Failed to start Textract job. Exiting.")
        return
    
    # Step 3: Wait for Textract job completion
    status = wait_for_textract_job(job_id)
    if status != "SUCCEEDED":
        print("Textract job did not succeed. Exiting.")
        return
    
    # Step 4: Retrieve Textract results
    results = retrieve_textract_results(job_id)
    
    # Step 5: Process and save extracted content
    process_textract_results(results, output_dir)

# Example usage
pdf_path = "./data/model_whitepaper.pdf"
bucket_name = "your-s3-bucket-name"
s3_key = "documents/model_whitepaper.pdf"
output_dir = "./extracted_content"

extract_content_from_pdf(pdf_path, bucket_name, s3_key, output_dir)
















import boto3
import time
import json
import os

def upload_pdf_to_s3(pdf_path, bucket_name, s3_key):
    """
    Uploads the PDF to an S3 bucket for processing by Textract.
    """
    s3 = boto3.client('s3')
    try:
        print(f"Uploading {pdf_path} to S3 bucket {bucket_name}...")
        s3.upload_file(pdf_path, bucket_name, s3_key)
        print(f"File uploaded successfully to {bucket_name}/{s3_key}.")
        return True
    except Exception as e:
        print(f"Error uploading file to S3: {str(e)}")
        return False

def start_textract_job(bucket_name, s3_key):
    """
    Starts a Textract job to analyze the PDF for text, tables, and forms.
    """
    textract = boto3.client('textract')
    try:
        print("Starting Textract job...")
        response = textract.start_document_analysis(
            DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': s3_key}},
            FeatureTypes=['TABLES', 'FORMS']  # Extract tables and forms
        )
        job_id = response['JobId']
        print(f"Textract Job ID: {job_id}")
        return job_id
    except Exception as e:
        print(f"Error starting Textract job: {str(e)}")
        return None

def wait_for_textract_job(job_id):
    """
    Waits for the Textract job to complete and returns the status.
    """
    textract = boto3.client('textract')
    while True:
        try:
            response = textract.get_document_analysis(JobId=job_id)
            status = response['JobStatus']
            if status in ['SUCCEEDED', 'FAILED']:
                print(f"Textract job completed with status: {status}")
                return status
            print(f"Job status: {status}. Waiting...")
            time.sleep(5)
        except Exception as e:
            print(f"Error checking Textract job status: {str(e)}")
            return None

def retrieve_textract_results(job_id):
    """
    Retrieves Textract results after the job is completed.
    Handles pagination for large documents.
    """
    textract = boto3.client('textract')
    results = []
    
    try:
        response = textract.get_document_analysis(JobId=job_id)
        results.append(response)
        
        next_token = response.get('NextToken')
        while next_token:
            response = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
            results.append(response)
            next_token = response.get('NextToken')
        
        print("Textract results retrieved successfully.")
        return results
    except Exception as e:
        print(f"Error retrieving Textract results: {str(e)}")
        return None

def process_textract_results(results, output_dir):
    """
    Processes Textract results to extract text, tables, and forms.
    Saves extracted data into separate files for downstream analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize containers for extracted data
    extracted_text = ""
    tables = []
    
    # Process each result block
    for result in results:
        blocks = result['Blocks']
        
        for block in blocks:
            block_type = block['BlockType']
            
            if block_type == 'LINE':  # Extract text lines
                extracted_text += block['Text'] + "\n"
            
            elif block_type == 'TABLE':  # Extract table data
                table_data = []
                cells = [cell for cell in blocks if cell['BlockType'] == 'CELL']
                for cell in cells:
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    text = cell.get('Text', '')
                    table_data.append((row_index, col_index, text))
                
                # Convert table data into a structured format (CSV-like)
                max_row_index = max([cell[0] for cell in table_data])
                max_col_index = max([cell[1] for cell in table_data])
                
                table_matrix = [["" for _ in range(max_col_index)] for _ in range(max_row_index)]
                for row_idx, col_idx, text in table_data:
                    table_matrix[row_idx-1][col_idx-1] = text
                
                tables.append(table_matrix)
    
    # Save extracted text
    with open(os.path.join(output_dir, "extracted_text.txt"), "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    # Save extracted tables as CSV files
    for i, table in enumerate(tables):
        table_path = os.path.join(output_dir, f"table_{i+1}.csv")
        with open(table_path, "w", encoding="utf-8") as f:
            for row in table:
                f.write(",".join(row) + "\n")
    
    print("Extracted content saved successfully.")
    
# Main function to execute the enhanced step 1 workflow
def extract_content_from_pdf(pdf_path, bucket_name, s3_key, output_dir):
    """
    Executes the full workflow of uploading a PDF to S3,
    processing it with Textract, and saving extracted content locally.
    """
    # Step 1: Upload the PDF to S3
    if not upload_pdf_to_s3(pdf_path, bucket_name, s3_key):
        print("Failed to upload PDF to S3. Exiting.")
        return
    
    # Step 2: Start Textract job
    job_id = start_textract_job(bucket_name, s3_key)
    if not job_id:
        print("Failed to start Textract job. Exiting.")
        return
    
    # Step 3: Wait for Textract job completion
    status = wait_for_textract_job(job_id)
    if status != "SUCCEEDED":
        print("Textract job did not succeed. Exiting.")
        return
    
    # Step 4: Retrieve Textract results
    results = retrieve_textract_results(job_id)
    
    # Step 5: Process and save extracted content
    process_textract_results(results, output_dir)

# Example usage
pdf_path = "./data/model_whitepaper.pdf"
bucket_name = "your-s3-bucket-name"
s3_key = "documents/model_whitepaper.pdf"
output_dir = "./extracted_content"

extract_content_from_pdf(pdf_path, bucket_name, s3_key, output_dir)












#########
import fitz  # PyMuPDF
import tabula
import pytesseract
from PIL import Image
import os
import json

def extract_text_from_pdf(pdf_path, output_dir):
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    Saves extracted text to a file.
    """
    os.makedirs(output_dir, exist_ok=True)
    extracted_text = ""
    
    # Open the PDF with PyMuPDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Processing {total_pages} pages from {pdf_path}...")
    
    for page_num in range(total_pages):
        page = doc[page_num]
        page_text = page.get_text()
        extracted_text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
        
        # Save individual page text for debugging/inspection
        with open(f"{output_dir}/page_{page_num + 1:03d}.txt", "w", encoding="utf-8") as f:
            f.write(page_text)
    
    # Save full extracted text
    full_text_path = os.path.join(output_dir, "extracted_text.txt")
    with open(full_text_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    print(f"Extracted text saved to {full_text_path}")
    return extracted_text

def extract_tables_from_pdf(pdf_path, output_dir):
    """
    Extracts tables from a PDF file using Tabula.
    Saves each table as a CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Extract tables from all pages
        tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True, lattice=True)
        
        if not tables:
            print("No tables found in the document.")
            return []
        
        table_files = []
        for i, table in enumerate(tables):
            if not table.empty:
                table_file = os.path.join(output_dir, f"table_{i + 1}.csv")
                table.to_csv(table_file, index=False)
                table_files.append(table_file)
                print(f"Table {i + 1} saved to {table_file}")
        
        return table_files
    except Exception as e:
        print(f"Error extracting tables: {e}")
        return []

def extract_images_from_pdf(pdf_path, output_dir):
    """
    Extracts images from a PDF file using PyMuPDF (fitz).
    Applies OCR to images if needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    image_files = []
    
    for page_num in range(total_pages):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image file
            image_file = os.path.join(output_dir, f"page_{page_num + 1}_image_{img_index + 1}.{image_ext}")
            with open(image_file, "wb") as f:
                f.write(image_bytes)
            
            image_files.append(image_file)
            
            # Apply OCR to the image if needed
            try:
                img = Image.open(image_file)
                ocr_text = pytesseract.image_to_string(img)
                
                if ocr_text.strip():
                    ocr_file = image_file.replace(f".{image_ext}", "_ocr.txt")
                    with open(ocr_file, "w", encoding="utf-8") as f:
                        f.write(ocr_text)
                    print(f"OCR applied to {image_file}, text saved to {ocr_file}")
            except Exception as e:
                print(f"Error applying OCR to {image_file}: {e}")
    
    print(f"Extracted {len(image_files)} images.")
    return image_files

def process_local_pdf(pdf_path, output_dir):
    """
    Full workflow to process a local PDF file.
    Extracts text, tables, and images.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\nStep 1: Extracting text...")
    extracted_text = extract_text_from_pdf(pdf_path, os.path.join(output_dir, "text"))

    print("\nStep 2: Extracting tables...")
    table_files = extract_tables_from_pdf(pdf_path, os.path.join(output_dir, "tables"))

    print("\nStep 3: Extracting images...")
    image_files = extract_images_from_pdf(pdf_path, os.path.join(output_dir, "images"))

    # Save metadata about extracted content
    metadata = {
        "pdf_path": pdf_path,
        "total_pages": len(fitz.open(pdf_path)),
        "text_file": os.path.join(output_dir, "text/extracted_text.txt"),
        "table_files": table_files,
        "image_files": image_files,
    }

    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nMetadata saved to {metadata_file}")
    
# Example usage
pdf_path = "./data/model_whitepaper.pdf"
output_dir = "./processed_content"

process_local_pdf(pdf_path, output_dir)


















import boto3
import time
import json
import os

def upload_pdf_to_s3(pdf_path, bucket_name, s3_key):
    """
    Uploads the PDF to an S3 bucket for processing by Textract.
    """
    s3 = boto3.client('s3')
    try:
        print(f"Uploading {pdf_path} to S3 bucket {bucket_name}...")
        s3.upload_file(pdf_path, bucket_name, s3_key)
        print(f"File uploaded successfully to {bucket_name}/{s3_key}.")
        return True
    except Exception as e:
        print(f"Error uploading file to S3: {str(e)}")
        return False

def start_textract_job(bucket_name, s3_key):
    """
    Starts a Textract job to analyze the PDF for text, tables, and forms.
    """
    textract = boto3.client('textract')
    try:
        print("Starting Textract job...")
        response = textract.start_document_analysis(
            DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': s3_key}},
            FeatureTypes=['TABLES', 'FORMS']  # Extract tables and forms
        )
        job_id = response['JobId']
        print(f"Textract Job ID: {job_id}")
        return job_id
    except Exception as e:
        print(f"Error starting Textract job: {str(e)}")
        return None

def wait_for_textract_job(job_id):
    """
    Waits for the Textract job to complete and returns the status.
    """
    textract = boto3.client('textract')
    while True:
        try:
            response = textract.get_document_analysis(JobId=job_id)
            status = response['JobStatus']
            if status in ['SUCCEEDED', 'FAILED']:
                print(f"Textract job completed with status: {status}")
                return status
            print(f"Job status: {status}. Waiting...")
            time.sleep(5)
        except Exception as e:
            print(f"Error checking Textract job status: {str(e)}")
            return None

def retrieve_textract_results(job_id):
    """
    Retrieves Textract results after the job is completed.
    Handles pagination for large documents.
    """
    textract = boto3.client('textract')
    results = []
    
    try:
        response = textract.get_document_analysis(JobId=job_id)
        results.append(response)
        
        next_token = response.get('NextToken')
        while next_token:
            response = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
            results.append(response)
            next_token = response.get('NextToken')
        
        print("Textract results retrieved successfully.")
        return results
    except Exception as e:
        print(f"Error retrieving Textract results: {str(e)}")
        return None

def process_textract_results(results, output_dir):
    """
    Processes Textract results to extract text, tables, and forms.
    Saves extracted data into separate files for downstream analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize containers for extracted data
    extracted_text = ""
    tables = []
    
    # Process each result block
    for result in results:
        blocks = result['Blocks']
        
        for block in blocks:
            block_type = block['BlockType']
            
            if block_type == 'LINE':  # Extract text lines
                extracted_text += block['Text'] + "\n"
            
            elif block_type == 'TABLE':  # Extract table data
                table_data = []
                cells = [cell for cell in blocks if cell['BlockType'] == 'CELL']
                for cell in cells:
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    text = cell.get('Text', '')
                    table_data.append((row_index, col_index, text))
                
                # Convert table data into a structured format (CSV-like)
                max_row_index = max([cell[0] for cell in table_data])
                max_col_index = max([cell[1] for cell in table_data])
                
                table_matrix = [["" for _ in range(max_col_index)] for _ in range(max_row_index)]
                for row_idx, col_idx, text in table_data:
                    table_matrix[row_idx-1][col_idx-1] = text
                
                tables.append(table_matrix)
    
    # Save extracted text
    with open(os.path.join(output_dir, "extracted_text.txt"), "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    # Save extracted tables as CSV files
    for i, table in enumerate(tables):
        table_path = os.path.join(output_dir, f"table_{i+1}.csv")
        with open(table_path, "w", encoding="utf-8") as f:
            for row in table:
                f.write(",".join(row) + "\n")
    
    print("Extracted content saved successfully.")
    
# Main function to execute the enhanced step 1 workflow
def extract_content_from_pdf(pdf_path, bucket_name, s3_key, output_dir):
    """
    Executes the full workflow of uploading a PDF to S3,
    processing it with Textract, and saving extracted content locally.
    """
    # Step 1: Upload the PDF to S3
    if not upload_pdf_to_s3(pdf_path, bucket_name, s3_key):
        print("Failed to upload PDF to S3. Exiting.")
        return
    
    # Step 2: Start Textract job
    job_id = start_textract_job(bucket_name, s3_key)
    if not job_id:
        print("Failed to start Textract job. Exiting.")
        return
    
    # Step 3: Wait for Textract job completion
    status = wait_for_textract_job(job_id)
    if status != "SUCCEEDED":
        print("Textract job did not succeed. Exiting.")
        return
    
    # Step 4: Retrieve Textract results
    results = retrieve_textract_results(job_id)
    
    # Step 5: Process and save extracted content
    process_textract_results(results, output_dir)

# Example usage
pdf_path = "./data/model_whitepaper.pdf"
bucket_name = "your-s3-bucket-name"
s3_key = "documents/model_whitepaper.pdf"
output_dir = "./extracted_content"

extract_content_from_pdf(pdf_path, bucket_name, s3_key, output_dir)
