from textractcaller.t_call import call_textract, Textract_Features
from textractprettyprinter.t_pretty_print import get_text_from_layout_json
import boto3
import tempfile
import os
import pdfplumber  # Import pdfplumber
import traceback
from botocore.exceptions import ClientError
import re
import json

def clean_table(table):
    """
    Cleans and normalizes a table extracted by pdfplumber.
    """
    cleaned_table = []
    for row in table:
        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
        cleaned_table.append(cleaned_row)
    return cleaned_table

def calculate_table_confidence(table):
    """
    Calculates a confidence score for the extracted table (a simple example).
    """
    total_cells = 0
    empty_cells = 0
    for row in table:
        for cell in row:
            total_cells += 1
            if not cell:
                empty_cells += 1

    if total_cells == 0:
        return 0  # Avoid division by zero

    confidence = 1 - (empty_cells / total_cells)
    return confidence

def extract_tables_from_pdf(pdf_path):
    """
    Extracts tables from a PDF document using pdfplumber.
    Handles potential errors during PDF processing.
    Returns a list of (table, confidence) tuples.
    """
    tables_with_confidence = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    for table in page_tables:
                        cleaned_table = clean_table(table)
                        confidence = calculate_table_confidence(cleaned_table)
                        tables_with_confidence.append((cleaned_table, confidence))
    except Exception as e:
        print(f"Error extracting tables: {e}")
        return None
    return tables_with_confidence

def download_s3_file(bucket_name, object_key, local_file_path):
    """
    Downloads a file from S3 to a local file path.
    """
    try:
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, object_key, local_file_path)
        return True
    except ClientError as e:
        print(f"Error downloading S3 file: {e}")
        return False

def extract_text_from_pdf(local_file_path, textract_client):
    """
    Extracts text from a PDF using Amazon Textract.
    """
    try:
        response = call_textract(
            input_document=local_file_path,
            features=[Textract_Features.LAYOUT, Textract_Features.TABLES],  # Include LAYOUT
            textract_client=textract_client,
            #LanguageCode='en',  # Specify language (e.g., 'en' for English)
            #OrientationCorrection='ROTATE_0'  # Specify orientation (e.g., 'ROTATE_0', 'ROTATE_90')
        )

        extracted_text = get_text_from_layout_json(
            textract_json=response,
            exclude_page_header=True,
            exclude_page_footer=True,
            exclude_page_number=True
        )
        return extracted_text
    except Exception as e:
        print(f"Error extracting text with Textract: {e}")
        traceback.print_exc()
        return None

def generate_table_schema(table_data):
    """
    Generates a table schema using LLM (placeholder).
    Replace with actual LLM call.
    """
    # This is a placeholder - replace with your LLM call here
    # The LLM should take the table data as input and return a schema
    # describing the columns and their data types.
    # Example:
    # schema = llm_client.generate_schema(table_data)
    # For now, let's return a simple static schema
    if table_data and len(table_data) > 0:
          num_columns = len(table_data[0])
          schema = {f"column_{i+1}": "text" for i in range(num_columns)}
    else:
        schema = {}  # Empty table, empty schema
    return schema

def extract_text_from_s3_pdf(bucket_name, object_key):
    """
    Downloads a PDF from S3, extracts tables and text (excluding headers/footers/page numbers),
    and returns the extracted data.
    """
    local_file_path = None  # Initialize to None
    try:
        s3_client = boto3.client('s3')
        textract_client = boto3.client('textract')

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            local_file_path = temp_file.name

            # Download the file from S3 to the temporary file
            if not download_s3_file(bucket_name, object_key, local_file_path):
                return None

            # Extract text using Textract
            extracted_text = extract_text_from_pdf(local_file_path, textract_client)

            # Extract tables using pdfplumber
            tables_with_confidence = extract_tables_from_pdf(local_file_path)

            if extracted_text is None or tables_with_confidence is None:
                print("Error: Text or Tables extractions is returning None")
                return None

            # Generate schema for each table
            tables_with_schema = []
            if tables_with_confidence:
                for table, confidence in tables_with_confidence:
                    table_schema = generate_table_schema(table)
                    tables_with_schema.append({"table": table, "confidence": confidence, "schema": table_schema})

            return {"text": extracted_text, "tables": tables_with_schema}

    except boto3.exceptions.S3.NoSuchBucket:
        print(f"Error: Bucket not found: {bucket_name}")
        return None

    except s3_client.exceptions.NoSuchKey:
        print(f"Error: Object not found: s3://{bucket_name}/{object_key}")
        return None
    except ClientError as e:
        print(f"AWS ClientError: {e}")
        return None

    except Exception as e:
        print(f"Error processing document: {e}")
        traceback.print_exc()  # Print the traceback
        return None

    finally:
        # Ensure the temporary file is deleted
        if local_file_path and os.path.exists(local_file_path):
            os.remove(local_file_path)


if __name__ == "__main__":
    bucket_name = "your-s3-bucket-name"  # Replace with your bucket name
    object_key = "path/to/your/whitepaper.pdf"  # Replace with your object key

    extracted_data = extract_text_from_s3_pdf(bucket_name, object_key)

    if extracted_data:
        print("Extraction successful!")
        print("Extracted Text:")
        print(extracted_data["text"])
        print("Extracted Tables:")
        for table_data in extracted_data["tables"]:
            print(f"Table: {table_data['table']}, Confidence: {table_data['confidence']}, Schema: {table_data['schema']}")
            # Convert table data to JSON format for printing
            table_json = json.dumps(table_data['table'], indent=4)
            print(f"Table (JSON):\n{table_json}")
    else:
        print("Text extraction failed.")
