import boto3
import json
import time
import concurrent.futures

def chunk_text(text, chunk_size=12000):
    """
    Splits text into optimized chunks while ensuring logical section grouping.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def process_chunk(bedrock, chunk, chunk_index):
    """
    Sends a single chunk to Claude 3 for analysis.
    """
    print(f"Processing chunk {chunk_index}...")
    
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
    
    **Document Content:**
    {chunk[:9000]}  # Ensure token limit safety
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
            print(f"Attempt {attempt+1} failed for chunk {chunk_index}: {e}")
            time.sleep(2)
    return {"chunk": chunk_index, "insights": "Error processing chunk"}

def analyze_text_with_claude3(text_file):
    """
    Processes extracted text using Claude 3 via AWS Bedrock in parallel to speed up processing.
    
    :param text_file: Path to the extracted text file.
    :return: Extracted model details stored as a knowledge base.
    """
    try:
        # Read extracted text
        with open(text_file, "r", encoding="utf-8") as f:
            extracted_text = f.read()
        
        # Initialize Bedrock client
        bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
        
        # Split text into chunks (optimized at 12,000 characters per chunk)
        text_chunks = chunk_text(extracted_text, chunk_size=12000)
        
        extracted_insights = []
        max_workers = min(10, len(text_chunks))  # Limit concurrency based on number of chunks
        
        # Process chunks in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {executor.submit(process_chunk, bedrock, chunk, i+1): i+1 for i, chunk in enumerate(text_chunks)}
            for future in concurrent.futures.as_completed(future_to_chunk):
                extracted_insights.append(future.result())
        
        # Sort results to maintain order
        extracted_insights.sort(key=lambda x: x["chunk"])
        
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
extracted_text_file = "extracted_text.txt"
analyze_text_with_claude3(extracted_text_file)
























###TEST
import boto3
import json
import time

def chunk_text(text, chunk_size=8000, max_chunks=None):
    """
    Splits text into smaller chunks to handle large documents efficiently.
    Allows limiting the number of chunks for testing purposes.
    """
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks if max_chunks is None else chunks[:max_chunks]

def analyze_text_with_claude3(text_file, test_mode=False, max_test_chunks=2):
    """
    Processes extracted text using Claude 3 via AWS Bedrock to generate structured insights
    and stores it in a knowledge base for question answering.
    Allows a test mode that limits the number of chunks processed.
    
    :param text_file: Path to the extracted text file.
    :param test_mode: If True, limits processing to a small number of chunks for testing.
    :param max_test_chunks: Maximum number of chunks to process in test mode.
    :return: Extracted model details stored as a knowledge base.
    """
    try:
        # Read extracted text
        with open(text_file, "r", encoding="utf-8") as f:
            extracted_text = f.read()
        
        # Initialize Bedrock client
        bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')  # Change region if needed
        
        # Split text into chunks and limit if in test mode
        text_chunks = chunk_text(extracted_text, max_chunks=max_test_chunks if test_mode else None)
        
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
# Set test_mode=True to process only a few chunks for testing
analyze_text_with_claude3(extracted_text_file, test_mode=True)




###
import PyPDF2
import pdfplumber
import csv

def extract_text_from_pdf(pdf_path, output_txt_path):
    """
    Extracts text from a PDF and saves it to a text file.
    Also extracts tabular data using pdfplumber and saves it as a CSV.
    
    :param pdf_path: Path to the PDF file.
    :param output_txt_path: Path to save extracted text.
    """
    try:
        # Extract text using PyPDF2
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text successfully extracted and saved to {output_txt_path}")
        
        # Extract tables using pdfplumber
        table_data = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_table()
                if tables:
                    table_data.extend(tables)
        
        # Save extracted tables to CSV
        if table_data:
            csv_output_path = output_txt_path.replace(".txt", ".csv")
            with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(table_data)
            print(f"Table data extracted and saved to {csv_output_path}")
        else:
            print("No tabular data found in the PDF.")
        
    except Exception as e:
        print(f"Error extracting data: {e}")

# Example usage
pdf_file = "your_model_whitepaper.pdf"  # Replace with your actual file
output_txt_file = "extracted_text.txt"
extract_text_from_pdf(pdf_file, output_txt_file)




#############################################

import PyPDF2
import pdfplumber
import csv

def extract_text_from_pdf(pdf_path, output_txt_path):
    """
    Extracts text from a PDF and saves it to a text file.
    Also extracts tabular data using pdfplumber and saves it as a CSV.
    
    :param pdf_path: Path to the PDF file.
    :param output_txt_path: Path to save extracted text.
    """
    try:
        # Extract text using PyPDF2
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text successfully extracted and saved to {output_txt_path}")
        
        # Extract tables using pdfplumber
        table_data = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_table()
                if tables:
                    table_data.extend(tables)
        
        # Save extracted tables to CSV
        if table_data:
            csv_output_path = output_txt_path.replace(".txt", ".csv")
            with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(table_data)
            print(f"Table data extracted and saved to {csv_output_path}")
        else:
            print("No tabular data found in the PDF.")
        
    except Exception as e:
        print(f"Error extracting data: {e}")

# Example usage
pdf_file = "your_model_whitepaper.pdf"  # Replace with your actual file
output_txt_file = "extracted_text.txt"
extract_text_from_pdf(pdf_file, output_txt_file)





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
            
            - Model Name
            - Model Version
            - Model Purpose & Scope
            - Model Inputs & Features
            - Model Processing & Calculations
            - Model Outputs & Usage
            - Performance Metrics & Evaluation
            - Assumptions & Limitations
            - Compliance & Regulatory Considerations
            - Explainability & Interpretability
            - Model Risk & Bias Considerations
            - Summary of Key Findings
            
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
