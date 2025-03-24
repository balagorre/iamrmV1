import boto3
import json
import re
import logging
import os
from textractcaller.t_call import call_textract, Textract_Features
from textractprettyprinter.t_pretty_print import get_text_from_layout_json, convert_table_to_list
from botocore.exceptions import ClientError

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def call_textract_with_s3_url(bucket_name: str, object_key: str) -> dict:
    """
    Calls Amazon Textract using an S3 object URL with LAYOUT and TABLES features.
    """
    try:
        textract_client = boto3.client('textract')
        s3_object_url = f"s3://{bucket_name}/{object_key}"
        logger.info(f"Calling Textract on: {s3_object_url}")

        response = call_textract(
            input_document=s3_object_url,
            features=[Textract_Features.LAYOUT, Textract_Features.TABLES],
            boto3_textract_client=textract_client,
        )

        if not response or "Blocks" not in response:
            raise ValueError("Textract response is empty or missing 'Blocks'.")

        logger.info("Textract call successful.")
        return response

    except ClientError as e:
        logger.exception("ClientError during Textract call.")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error calling Textract: {e}")
        raise


def filter_low_confidence_blocks(blocks: list, threshold: float = 85.0) -> list:
    """Filters blocks based on confidence score."""
    return [block for block in blocks if block.get('Confidence', 0) >= threshold]


def clean_extracted_text(text: str) -> str:
    """Cleans extracted text by removing unwanted characters and formatting."""
    return re.sub(r'\s+', ' ', text).strip()


def extract_text_from_textract_response(textract_response: dict) -> str:
    """Extracts clean text using Textract layout parser."""
    try:
        logger.info("Extracting text from Textract response...")

        if 'Blocks' not in textract_response or not isinstance(textract_response['Blocks'], list):
            raise ValueError("Invalid Textract response: missing or malformed 'Blocks'.")

        filtered_blocks = filter_low_confidence_blocks(textract_response['Blocks'])

        extracted_text = get_text_from_layout_json(
            textract_json={"Blocks": filtered_blocks},
            exclude_page_header=True,
            exclude_page_footer=True,
            exclude_page_number=True,
        )

        cleaned_text = clean_extracted_text(extracted_text)
        logger.info("Text extraction successful.")
        return cleaned_text

    except Exception as e:
        logger.exception(f"Error extracting text from Textract response: {e}")
        raise


def extract_tables_from_textract_response(textract_response: dict) -> list:
    """Extracts tables using textractprettyprinter."""
    try:
        logger.info("Extracting tables from Textract response...")

        table_data = convert_table_to_list(textract_response)

        if not isinstance(table_data, list):
            logger.warning(f"Unexpected table output format: {type(table_data)}")
            return []

        logger.info(f"Extracted {len(table_data)} tables.")
        return table_data

    except Exception as e:
        logger.exception(f"Error extracting tables: {e}")
        return []  # Return empty list if table extraction fails


def save_output_to_file(data: dict, output_file_path: str) -> None:
    """Saves extracted text and tables to JSON."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4)
        logger.info(f"Output saved to {output_file_path}")
    except Exception as e:
        logger.exception(f"Failed to save output: {e}")
        raise


def process_s3_pdf(bucket_name: str, object_key: str, output_file_path: str) -> bool:
    """
    Full pipeline: Call Textract, extract text + tables, and save to file.
    """
    try:
        textract_response = call_textract_with_s3_url(bucket_name, object_key)

        extracted_text = extract_text_from_textract_response(textract_response)
        tables = extract_tables_from_textract_response(textract_response)

        save_output_to_file({
            "text": extracted_text,
            "tables": tables
        }, output_file_path)

        logger.info("PDF processing completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Failed to process PDF from S3: {e}")
        return False


# --- Entry point ---
if __name__ == "__main__":
    # Replace with actual S3 bucket and file
    bucket_name = os.environ.get("BUCKET_NAME", "your-s3-bucket-name")
    object_key = os.environ.get("OBJECT_KEY", "path/to/your/whitepaper.pdf")
    output_file = os.environ.get("OUTPUT_FILE", "processed_output.json")

    success = process_s3_pdf(bucket_name, object_key, output_file)
    if success:
        logger.info("✔️ Extraction complete.")
    else:
        logger.error("❌ Extraction failed.")
