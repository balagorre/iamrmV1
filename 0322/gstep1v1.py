import boto3
import io
import json
from botocore.exceptions import ClientError
from textractcaller.t_call import call_textract, Textract_Features
from textractprettyprinter.t_pretty_print import get_text_from_layout_json
from textractresponseparser import response_parser
import logging
from typing import Dict

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_s3_file_to_memory(bucket_name: str, object_key: str) -> io.BytesIO:
    """
    Downloads a file from S3 into an in-memory BytesIO object.

    Args:
        bucket_name (str): The S3 bucket name.
        object_key (str): The S3 object key.

    Returns:
        io.BytesIO: The file content as an in-memory bytes object.
    """
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

def extract_text_and_tables(textract_response: Dict) -> Dict:
    """
    Extracts text and tables from a Textract response.

    Args:
        textract_response (Dict): The JSON response from Amazon Textract.

    Returns:
        Dict: A dictionary containing extracted text and tables.
    """
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

        # Extract tables using amazon-textract-response-parser
        parser = response_parser.TextractResponseParser(textract_response)
        extracted_tables = parser.get_all_tables_as_list()

        logger.info(f"Extracted {len(extracted_tables)} tables.")
        return {"text": extracted_text.strip(), "tables": extracted_tables}
    except Exception as e:
        logger.error(f"Error extracting text/tables: {e}")
        raise

def save_output_to_file(data: Dict, output_file_path: str) -> None:
    """
    Saves extracted data to a JSON file.

    Args:
        data (Dict): The extracted data.
        output_file_path (str): Path to save the JSON file.
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4)
        logger.info(f"Saved output to {output_file_path}")
    except Exception as e:
        logger.error(f"Error saving output: {e}")
        raise

def process_pdf_from_s3(bucket_name: str, object_key: str, output_file_path: str) -> bool:
    """
    Downloads a PDF from S3, extracts text and tables using Amazon Textract,
    and saves the processed data to a local JSON file.

    Args:
        bucket_name (str): The S3 bucket name.
        object_key (str): The S3 object key.
        output_file_path (str): Path to save the processed JSON output.

    Returns:
        bool: True if processing is successful, False otherwise.
    """
    try:
        # Step 1: Download PDF from S3 into memory
        pdf_stream = download_s3_file_to_memory(bucket_name, object_key)

        # Step 2: Call Amazon Textract for text and table extraction
        textract_client = boto3.client('textract')
        textract_response = call_textract(
            input_document=pdf_stream.getvalue(),
            features=[Textract_Features.LAYOUT, Textract_Features.TABLES],
            textract_client=textract_client,
            # Uncomment below if needed for specific configurations
            # LanguageCode='en',
            # OrientationCorrection='ROTATE_0'
        )

        # Step 3: Extract text and tables from the Textract response
        extracted_data = extract_text_and_tables(textract_response)

        # Validate extracted data before saving
        if not extracted_data.get("text") or not isinstance(extracted_data["tables"], list):
            raise ValueError("Extracted data is invalid.")

        # Step 4: Save the processed output locally
        save_output_to_file(extracted_data, output_file_path)

        logger.info("PDF processing completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Error processing PDF from S3: {e}")
        return False

if __name__ == "__main__":
    # Replace these values with your actual S3 bucket and object details
    bucket_name = "your-s3-bucket-name"
    object_key = "path/to/your/whitepaper.pdf"
    
    # Output path for saving the processed data locally
    output_json_file = "processed_output.json"

    success = process_pdf_from_s3(bucket_name, object_key, output_json_file)

