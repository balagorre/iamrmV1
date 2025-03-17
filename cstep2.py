import boto3
import json
import time
import concurrent.futures
import csv
import datetime

def chunk_text(text, chunk_size=12000):
    """
    Splits text into optimized chunks while ensuring logical section grouping.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def read_csv_data(csv_file):
    """
    Reads extracted tables from CSV and converts them into a Markdown-style table format.
    Preserves row-column structure instead of simple JSON representation.
    """
    table_data = []
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, [])  # Extract headers
            table_data.append(" | ".join(headers))
            table_data.append(" | ".join(["---"] * len(headers)))  # Markdown separator
            for row in reader:
                table_data.append(" | ".join(row))
    except Exception as e:
        log_error(f"Error reading CSV file: {e}")
    return "\n".join(table_data) if table_data else "No structured table data available."

def log_error(error_message):
    """
    Logs error messages to an external error log file.
    """
    with open("error_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now()} - {error_message}\n")

def process_chunk(bedrock, chunk, csv_data, chunk_index):
    """
    Sends a single chunk to Claude 3 for analysis, including structured CSV data.
    """
    print(f"Processing chunk {chunk_index}...")
    
    prompt = f"""
    You are an AI assistant specialized in analyzing complex financial, regulatory, and AI model documentation.
    Below is extracted model documentation along with structured tabular data.
    
    ## **Structured Table Data (Markdown Format):**
    {csv_data}
    
    ## **Document Content:**
    {chunk[:9000]}  # Ensure token limit safety
    
    **Extract and analyze the following aspects:**
    - **Model Overview:** Name, Version, Purpose & Scope
    - **Model Inputs:** Data sources, transformations applied
    - **Processing & Calculations:** Key computations, algorithms, ML techniques
    - **Model Outputs & Performance:** How outputs are used, metrics, benchmarks
    - **Assumptions & Limitations:** Risks, biases, known issues
    - **Regulatory Compliance:** Basel, GDPR, risk controls
    - **Explainability & Bias:** SHAP, LIME, interpretability methods
    - **Summary of Key Findings:** Highlight inconsistencies, missing details
    """
    
    payload = {
        "prompt": prompt,
        "max_tokens": min(2000, 4096 - len(chunk)//4),  # Adjust tokens dynamically
        "temperature": 0.5
    }
    
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            response = bedrock.invoke_model(
                body=json.dumps(payload),
                modelId='anthropic.claude-3-sonnet-2024-02-29',
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response['body'].read().decode('utf-8'))
            insights = response_body.get('completion', 'No response received')
            return {"chunk": chunk_index, "insights": insights}
        except Exception as e:
            log_error(f"Attempt {attempt+1} failed for chunk {chunk_index}: {e}")
            time.sleep(2)
    return {"chunk": chunk_index, "insights": "Error processing chunk"}

def analyze_text_with_claude3(text_file, csv_file):
    """
    Processes extracted text and structured tables using Claude 3 via AWS Bedrock.
    Includes execution time tracking and improved error handling.
    """
    start_time = time.time()
    try:
        # Read extracted text
        with open(text_file, "r", encoding="utf-8") as f:
            extracted_text = f.read()
        
        # Read structured table data
        csv_data = read_csv_data(csv_file)
        
        # Initialize Bedrock client
        bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
        
        # Split text into chunks (optimized at 12,000 characters per chunk)
        text_chunks = chunk_text(extracted_text, chunk_size=12000)
        
        extracted_insights = []
        max_workers = min(10, len(text_chunks))  # Limit concurrency based on number of chunks
        
        # Process chunks in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(process_chunk, bedrock, chunk, csv_data, i+1): i+1 for i, chunk in enumerate(text_chunks)}
            for future in concurrent.futures.as_completed(future_to_chunk):
                extracted_insights.append(future.result())
        
        # Sort results to maintain order
        extracted_insights.sort(key=lambda x: x["chunk"])
        
        # Save results as a structured knowledge base (JSON)
        kb_file = text_file.replace(".txt", "_knowledge_base.json")
        with open(kb_file, "w", encoding="utf-8") as f:
            json.dump(extracted_insights, f, indent=4)
        
        end_time = time.time()
        execution_time = str(datetime.timedelta(seconds=int(end_time - start_time)))
        print(f"Knowledge base saved to {kb_file}. Execution time: {execution_time}")
        return kb_file
    
    except Exception as e:
        log_error(f"Error analyzing text: {e}")
        return None

# Example usage
extracted_text_file = "extracted_text.txt"
extracted_csv_file = "extracted_text.csv"
analyze_text_with_claude3(extracted_text_file, extracted_csv_file)


































import boto3
import json
import time

def chunk_text(text, chunk_size=8000):
    """
    Splits text into smaller chunks to handle large documents efficiently.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def analyze_text_with_claude3(text_file):
    """
    Processes extracted text using Claude 3 via AWS Bedrock to generate structured insights
    and stores it in a knowledge base for question answering.
    
    :param text_file: Path to the extracted text file.
    :return: Extracted model details stored as a knowledge base.
    """
    try:
        # Read extracted text
        with open(text_file, "r", encoding="utf-8") as f:
            extracted_text = f.read()
        
        # Initialize Bedrock client
        bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')  # Change region if needed
        
        # Split text into chunks to avoid token limit issues
        text_chunks = chunk_text(extracted_text)
        
        extracted_insights = []
        
        for i, chunk in enumerate(text_chunks):
            print(f"Processing chunk {i+1}/{len(text_chunks)}...")
            
            # Define Claude 3 prompt
            prompt = f"""
            You are an AI assistant specialized in analyzing complex financial, regulatory, and AI model documentation.
            Analyze the following model whitepaper and extract structured information:
            
            1. **Model Overview:**
               - Model Name
               - Model Version
               - Model Purpose & Scope (Explain in detail)
            
            2. **Model Inputs:**
               - List all required inputs, including data sources, features, and transformations applied.
               - Explain the role of each input in the model.
               
            3. **Model Processing & Calculations:**
               - Detail the core computations, algorithms, and methodologies used.
               - Explain how the model processes the inputs and generates outputs.
               - Mention key mathematical formulas or machine learning techniques involved.
            
            4. **Model Outputs:**
               - List all outputs generated by the model.
               - Define how each output is used in business decisions.
            
            5. **Performance & Evaluation Metrics:**
               - Explain how the model’s accuracy, robustness, and effectiveness are measured.
               - List key performance metrics and benchmark results.
            
            6. **Assumptions & Limitations:**
               - Detail all underlying assumptions considered during model development.
               - Highlight known limitations and potential risks associated with the model.
            
            7. **Regulatory & Compliance Considerations:**
               - Identify any applicable regulations the model adheres to (e.g., Basel, SR 11-7, GDPR, etc.).
               - List risk controls and governance requirements.
            
            8. **Explainability & Interpretability:**
               - Describe how model decisions can be interpreted.
               - Mention interpretability techniques (e.g., SHAP, LIME, feature importance analysis).
            
            9. **Model Risk & Bias Considerations:**
               - Identify potential sources of model bias.
               - Describe risk mitigation strategies (e.g., fairness constraints, adversarial testing).
               - Explain how model drift and degradation are monitored over time.
            
            10. **Summary of Key Findings:**
               - Provide a concise summary of extracted details.
               - Highlight inconsistencies, gaps, or missing details within the document.
            
            **Important:** Ensure accuracy and completeness when analyzing structured sections. Prioritize factual information from the document, and avoid making assumptions beyond the given text.
            
            **Document Content:**
            {chunk}
            """
            
            payload = {
                "prompt": prompt,
                "max_tokens": 1500,
                "temperature": 0.5
            }
            
            # Invoke Claude 3 with retry logic
            retry_attempts = 3
            for attempt in range(retry_attempts):
                try:
                    response = bedrock.invoke_model(
                        body=json.dumps(payload),
                        modelId='anthropic.claude-3-sonnet-2024-02-29',  # Update model version if needed
                        accept='application/json',
                        contentType='application/json'
                    )
                    
                    response_body = json.loads(response['body'].read().decode('utf-8'))
                    insights = response_body.get('completion', 'No response received')
                    extracted_insights.append({"chunk": i+1, "insights": insights})
                    break  # Exit retry loop if successful
                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {e}")
                    time.sleep(2)  # Wait before retrying
            
        # Save results as a structured knowledge base (JSON)
        kb_file = text_file.replace(".txt", "_knowledge_base.json")
        with open(kb_file, "w", encoding="utf-8") as f:
            json.dump(extracted_insights, f, indent=4)
        
        print(f"Knowledge base saved to {kb_file}")
        return kb_file
    
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return None

# Example usage
extracted_text_file = "extracted_text.txt"  # Ensure this file exists from Step 1
analyze_text_with_claude3(extracted_text_file)
