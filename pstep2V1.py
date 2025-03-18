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

# Example usage
with open("./extracted_content/extracted_text.txt", "r", encoding="utf-8") as f:
    extracted_text = f.read()

chunks = chunk_text(extracted_text)
print(f"Total chunks created: {len(chunks)}")


def create_prompt_with_examples(chunk, chunk_id):
    """
    Creates a detailed prompt for analyzing a chunk of text using Claude with enhanced parameters.
    
    Args:
        chunk (str): A chunk of text to analyze.
        chunk_id (int): Unique identifier for the chunk.

    Returns:
        str: A formatted prompt string with detailed instructions and examples.
    """
    prompt = f"""
    You are an expert auditor reviewing a financial model whitepaper. Analyze the following content:

    {chunk}

    Tasks:
    1. Provide a summary of this content.
    2. Identify all model inputs and describe their purpose and format (e.g., numerical, categorical).
    3. Identify all model outputs and describe their purpose and format (e.g., numerical, categorical).
    4. List all key calculations performed by the model and explain their significance.
    5. Describe how model performance is evaluated based on metrics or indicators (e.g., accuracy, precision).
    6. Extract details about the solution specification (e.g., architecture, components).
    7. Summarize any testing conducted for the model (e.g., test cases, coverage).
    8. Highlight any reconciliation processes mentioned in this content (e.g., data validation or matching logic).

    Instructions:
    - If specific information is missing in this chunk, state "Not found in context."
    - Follow the exact format provided below for your response.
    - Ensure all fields are filled out accurately based on the content of the chunk.

    Example Response Format:

    Chunk ID: {chunk_id}

    Summary:
      This section describes how the model processes input data to generate predictions.

    Inputs:
      - input_1: Represents historical sales data used to predict future trends (numerical). Source: Sales database.
      - input_2: Indicates customer segmentation based on demographic data (categorical). Source: CRM system.

    Outputs:
      - output_1: Provides predicted sales figures for the next quarter (numerical). Source: Prediction engine.
      - output_2: Generates risk scores for customers based on historical behavior (numerical). Source: Risk module.

    Calculations:
      - Linear regression formula applied to sales data for trend prediction.
      - Clustering algorithm used for customer segmentation.

    Model Performance:
      - Accuracy: The model achieves 95% accuracy on historical sales predictions.
      - Precision: Precision is measured at 90% for high-risk customer identification.

    Solution Specification:
      - Architecture: The solution uses a microservices-based architecture with components for data ingestion, preprocessing, and prediction generation.
      - Components: Includes modules for feature engineering and real-time prediction serving.

    Testing Summary:
      - Test Case_1: Validates input data preprocessing logic.
      - Test Case_2: Ensures predictions align with expected trends under stable conditions.

    Reconciliation:
      - Matches predicted sales figures against actual sales data from previous quarters.
      - Validates customer risk scores against historical behavior patterns.

    Return your response in this exact format."""
    
    return prompt


import boto3
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)

def analyze_chunk_with_claude(chunk, chunk_id):
    """
    Analyzes a single chunk of text using Claude via AWS Bedrock with an enhanced prompt.
    
    Args:
        chunk (str): A chunk of text to analyze.
        chunk_id (int): Unique identifier for the chunk.

    Returns:
        dict: Parsed structured data from Claude's response or an error message.
    """
    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    prompt = create_prompt_with_examples(chunk, chunk_id)
    
    try:
        # Invoke Claude via AWS Bedrock
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
            })
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read().decode('utf-8'))
        response_text = response_body["content"]
        
        # Parse the plain-text response into structured data
        return parse_enhanced_response(response_text)
    
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_id}: {str(e)}")
        return {"chunk_id": chunk_id, "error": f"Failed to process chunk {chunk_id}: {str(e)}"}



def process_chunks_concurrently(chunks):
    """
    Processes all chunks concurrently using multi-threading.
    
    Args:
        chunks (list): List of text chunks to process.

    Returns:
        list: List of parsed results from all chunks.
    """
    results = []
    
    logging.info(f"Processing {len(chunks)} chunks concurrently...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks for each chunk
        futures = [
            executor.submit(analyze_chunk_with_claude, chunk, i + 1)
            for i, chunk in enumerate(chunks)
        ]
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"Error in thread execution: {str(e)}")
    
    logging.info("All chunks processed.")
    
    return results

# Example usage
results = process_chunks_concurrently(chunks)


def consolidate_results(results):
    """
    Consolidates results from all chunks into a single structured output.
    
    Args:
        results (list): List of results from individual chunks.

    Returns:
        dict: Consolidated results including summaries, inputs/outputs,
              calculations, model performance, solution specification,
              testing summary, reconciliation along with metadata about processing.
    """
    consolidated = {
        "summary": [],
        "inputs": [],
        "outputs": [],
        "calculations": [],
        "model_performance": [],
        "solution_specification": [],
        "testing_summary": [],
        "reconciliation": [],
        "metadata": {
            "total_chunks": len(results),
            "processed_chunks": 0,
            "failed_chunks": 0,
            "failed_chunk_ids": []
        }
    }
    
    for i, result in enumerate(results):
        # Validate that result is a dictionary
        if not isinstance(result, dict) or result.get("error"):
            consolidated["metadata"]["failed_chunks"] += 1
            consolidated["metadata"]["failed_chunk_ids"].append(result.get("chunk_id", f"Invalid result at index {i}"))
            continue
        
        # Track processed chunks
        consolidated["metadata"]["processed_chunks"] += 1
        
        # Consolidate summaries
        summary = result.get("summary", "").strip()
        if summary:
            consolidated["summary"].append(summary)
        
        # Deduplicate model inputs/outputs/assumptions/limitations
        for input_item in result.get("inputs", []):
            if not any(existing_input["name"] == input_item["name"] for existing_input in consolidated["inputs"]):
                consolidated["inputs"].append(input_item)
        
        for output_item in result.get("outputs", []):
            if not any(existing_output["name"] == output_item["name"] for existing_output in consolidated["outputs"]):
                consolidated["outputs"].append(output_item)
        
        for calculation in result.get("calculations", []):
            if calculation not in consolidated["calculations"]:
                consolidated["calculations"].append(calculation)
        
        for performance_metric in result.get("model_performance", []):
            if performance_metric not in consolidated["model_performance"]:
                consolidated["model_performance"].append(performance_metric)
        
        for spec_item in result.get("solution_specification", []):
            if spec_item not in consolidated["solution_specification"]:
                consolidated["solution_specification"].append(spec_item)
        
        for test_case in result.get("testing_summary", []):
            if test_case not in consolidated["testing_summary"]:
                consolidated["testing_summary"].append(test_case)
        
        for reconciliation_process in result.get("reconciliation", []):
            if reconciliation_process not in consolidated["reconciliation"]:
                consolidated["reconciliation"].append(reconciliation_process)
    
    return consolidated

# Example usage
final_results = consolidate_results(results)

# Save final results to file
import json
with open("./extracted_content/final_whitepaper_analysis.json", "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2)

print("Final whitepaper analysis saved.")
