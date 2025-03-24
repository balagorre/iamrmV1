import boto3
import json
import re
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

        # Call Textract API with LAYOUT and TABLES features enabled
        response = call_textract(
            input_document=s3_object_url,
            features=[Textract_Features.LAYOUT, Textract_Features.TABLES],
            boto3_textract_client=textract_client,
        )

        logger.info("Textract call successful.")
        return response

    except Exception as e:
        logger.error(f"Error calling Textract: {e}")
        raise

def filter_low_confidence_blocks(blocks: list, threshold: float = 90.0) -> list:
    """
    Filters blocks based on confidence score.

    Args:
        blocks (list): List of blocks from Textract response.
        threshold (float): Minimum confidence score to include a block.

    Returns:
        list: Filtered blocks with confidence >= threshold.
    """
    return [block for block in blocks if block.get('Confidence', 0) >= threshold]

def clean_extracted_text(text: str) -> str:
    """
    Cleans extracted text by removing unwanted characters and formatting.

    Args:
        text (str): The raw extracted text.

    Returns:
        str: Cleaned text.
    """
    # Example: Remove extra spaces and line breaks
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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

        # Validate that 'Blocks' exist in the response
        if 'Blocks' not in textract_response or not isinstance(textract_response['Blocks'], list):
            raise ValueError("Invalid Textract response: Missing 'Blocks' or incorrect format.")

        # Filter low-confidence blocks
        filtered_blocks = filter_low_confidence_blocks(textract_response['Blocks'])

        # Use pretty printer to extract text from filtered blocks
        extracted_text = get_text_from_layout_json(
            textract_json={"Blocks": filtered_blocks},
            exclude_page_header=True,
            exclude_page_footer=True,
            exclude_page_number=True,
        )

        # Clean extracted text before returning
        cleaned_text = clean_extracted_text(extracted_text)

        logger.info("Text extraction completed successfully.")
        return cleaned_text

    except Exception as e:
        logger.error(f"Error extracting text from Textract response: {e}")
        raise

def save_output_to_file(data: dict, output_file_path: str) -> None:
    """
    Saves extracted data to a JSON file.

    Args:
        data (dict): The extracted data.
        output_file_path (str): Path to save the JSON file.
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4)
        logger.info(f"Saved output to {output_file_path}")
    except Exception as e:
        logger.error(f"Error saving output: {e}")
        raise

def process_s3_pdf(bucket_name: str, object_key: str, output_file_path: str) -> bool:
    """
    Processes a PDF stored in S3 by calling Textract and extracting text.

    Args:
        bucket_name (str): The name of the S3 bucket.
        object_key (str): The key of the S3 object.
        output_file_path (str): Path to save the processed JSON output.

    Returns:
        bool: True if processing is successful, False otherwise.
    """
    try:
        # Step 1: Call Textract with the S3 URL
        textract_response = call_textract_with_s3_url(bucket_name, object_key)

        # Step 2: Extract text from the Textract response
        extracted_text = extract_text_from_textract_response(textract_response)

        # Step 3: Save processed output locally
        save_output_to_file({"text": extracted_text}, output_file_path)

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

    success = process_s3_pdf(bucket_name, object_key, output_json_file)

