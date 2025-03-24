import boto3
import json
import logging
from textractcaller.t_call import call_textract, Textract_Features
from textractprettyprinter.t_pretty_print import get_text_from_layout_json

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def call_textract_with_s3_url(bucket_name: str, object_key: str) -> dict:
    """
    Calls Amazon Textract using an S3 object URL.

    Args:
        bucket_name (str): The name of the S3 bucket.
        object_key (str): The key of the S3 object.

    Returns:
        dict: The Textract response.
    """
    try:
        # Create an Amazon Textract client
        textract_client = boto3.client('textract')

        # Generate the S3 object URL
        s3_object_url = f"s3://{bucket_name}/{object_key}"
        logger.info(f"Calling Textract with S3 URL: {s3_object_url}")

        # Call Textract API with LAYOUT feature enabled
        response = call_textract(
            input_document=s3_object_url,
            features=[Textract_Features.LAYOUT],
            textract_client=textract_client,
        )

        logger.info("Textract call successful.")
        return response

    except Exception as e:
        logger.error(f"Error calling Textract: {e}")
        raise

def extract_text_from_textract_response(textract_response: dict) -> str:
    """
    Extracts text from the Textract response using get_text_from_layout_json.

    Args:
        textract_response (dict): The JSON response from Amazon Textract.

    Returns:
        str: The extracted text.
    """
    try:
        logger.info("Extracting text from Textract response...")

        # Use pretty printer to extract text
        extracted_text = get_text_from_layout_json(
            textract_json=textract_response,
            exclude_page_header=True,
            exclude_page_footer=True,
            exclude_page_number=True,
        )

        logger.info("Text extraction completed successfully.")
        return extracted_text.strip()

    except Exception as e:
        logger.error(f"Error extracting text from Textract response: {e}")
        raise

def process_s3_pdf(bucket_name: str, object_key: str) -> None:
    """
    Processes a PDF stored in S3 by calling Textract and extracting text.

    Args:
        bucket_name (str): The name of the S3 bucket.
        object_key (str): The key of the S3 object.
    """
    try:
        # Step 1: Call Textract with the S3 URL
        textract_response = call_textract_with_s3_url(bucket_name, object_key)

        # Step 2: Extract text from the Textract response
        extracted_text = extract_text_from_textract_response(textract_response)

        # Step 3: Save extracted text locally
        output_file_path = "extracted_text.txt"
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            outfile.write(extracted_text)
        
        logger.info(f"Extracted text saved to {output_file_path}")

    except Exception as e:
        logger.error(f"Error processing PDF from S3: {e}")

if __name__ == "__main__":
    # Replace these values with your actual S3 bucket and object details
    bucket_name = "your-s3-bucket-name"
    object_key = "path/to/your/whitepaper.pdf"

    process_s3_pdf(bucket_name, object_key)
