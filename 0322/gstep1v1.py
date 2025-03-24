pip install amazon-textract-response-parser

from textractresponseparser import response_parser





import boto3
import io
import json
from botocore.exceptions import ClientError
from textractcaller.t_call import call_textract, Textract_Features
from textractprettyprinter.t_pretty_print import get_text_from_layout_json
from textractresponseparser.response_parser import parse  # Correctly importing the parser module

import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_s3_file_to_memory(bucket_name: str, object_key: str) -> io.BytesIO:
    """Downloads a file from S3 into an in-memory BytesIO object."""
    try:
        s3_client = boto3.client('s3')
        file_stream = io.BytesIO()
        s3_client.download_fileobj(bucket_name, object_key, file_stream)
        file_stream.seek(0)  # Reset stream position
        logger.info(f"Downloaded s3://{bucket_name}/{object_key} into memory.")
        return file_stream
    except ClientError as e:
        logger.error(f"Error downloading from S3: {e}")
        raise Exception(f"Failed to download file from S3: {e}")

def extract_text_and_tables(textract_response: dict) -> dict:
    """Extracts text and tables from a Textract response."""
    try:
        # Validate Textract response structure
        if not textract_response or 'Blocks' not in textract_response:
            raise ValueError("Invalid Textract response: Missing 'Blocks'.")

        # Extract text using pretty printer
        extracted_text = get_text_from_layout_json(
            textract_json=textract_response,
            exclude_page_header=True,
            exclude_page_footer=True,
            exclude_page_number=True,
        )

        # Extract tables using response parser
        parsed_document = parse(textract_response)
        extracted_tables = [table.to_list() for table in parsed_document.tables]

        logger.info(f"Extracted {len(extracted_tables)} tables.")
        return {"text": extracted_text.strip(), "tables": extracted_tables}
    except Exception as e:
        logger.error(f"Error extracting text/tables: {e}")
        raise

def save_output_to_file(data: dict, output_file_path: str) -> None:
    """Saves extracted data to a JSON file."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4)
        logger.info(f"Saved output to {output_file_path}")
    except Exception as e:
        logger.error(f"Error saving output: {e}")
        raise

def process_pdf_from_s3(bucket_name: str, object_key: str, output_file_path: str) -> bool:
    """Downloads a PDF from S3, extracts text and tables using Amazon Textract, and saves the processed data locally."""
    try:
        pdf_stream = download_s3_file_to_memory(bucket_name, object_key)

        textract_client = boto3.client('textract')
        textract_response = call_textract(
            input_document=pdf_stream.getvalue(),
            features=[Textract_Features.LAYOUT, Textract_Features.TABLES],
            textract_client=textract_client,
            # Uncomment below if needed for specific configurations
            # LanguageCode='en',
            # OrientationCorrection='ROTATE_0'
        )

        extracted_data = extract_text_and_tables(textract_response)

        if not extracted_data.get("text") or not isinstance(extracted_data["tables"], list):
            raise ValueError("Extracted data is invalid.")

        save_output_to_file(extracted_data, output_file_path)
        
        logger.info("PDF processing completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Error processing PDF from S3: {e}")
        return False

if __name__ == "__main__":
    bucket_name = "your-s3-bucket-name"
    object_key = "path/to/your/whitepaper.pdf"
    
    output_json_file = "processed_output.json"

    success = process_pdf_from_s3(bucket_name, object_key, output_json_file)

    if success:
        logger.info("PDF processing successful!")
    else:
        logger.error("PDF processing failed.")
