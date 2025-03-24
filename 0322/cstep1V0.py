import boto3
import json
import re
import logging
from textractcaller.t_call import call_textract, Textract_Features
from textractprettyprinter.t_pretty_print import get_text_from_layout_json, convert_table_to_list
from botocore.exceptions import ClientError

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def call_textract_with_bytes(bucket_name: str, object_key: str) -> dict:
    """
    Downloads a PDF file from S3 and calls Textract using bytes input.
    This ensures LAYOUT + TABLES features are honored properly.
    """
    try:
        s3 = boto3.client('s3')
        textract = boto3.client('textract')

        logger.info(f"Downloading s3://{bucket_name}/{object_key} into memory")
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        file_bytes = response['Body'].read()

        logger.info("Calling Textract with LAYOUT and TABLES features")
        textract_response = call_textract(
            input_document=file_bytes,
            features=[Textract_Features.LAYOUT, Textract_Features.TABLES],
            boto3_textract_client=textract,
        )

        if not textract_response or 'Blocks' not in textract_response:
            raise ValueError("Textract returned an empty or invalid response.")

        return textract_response

    except Exception as e:
        logger.exception(f"Textract call failed: {e}")
        raise

def filter_low_confidence_blocks(blocks, threshold=85.0):
    return [block for block in blocks if block.get('Confidence', 0) >= threshold]

def clean_extracted_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def extract_text(textract_response: dict) -> str:
    try:
        logger.info("Extracting clean text from Textract response")
        blocks = textract_response.get("Blocks", [])
        filtered = filter_low_confidence_blocks(blocks)
        extracted_text = get_text_from_layout_json(
            textract_json={"Blocks": filtered},
            exclude_page_header=True,
            exclude_page_footer=True,
            exclude_page_number=True
        )
        return clean_extracted_text(extracted_text)
    except Exception as e:
        logger.exception(f"Text extraction failed: {e}")
        return ""

def extract_tables(textract_response: dict) -> list:
    try:
        logger.info("Extracting tables from Textract response")
        tables = convert_table_to_list(textract_response)
        return tables if isinstance(tables, list) else []
    except Exception as e:
        logger.warning(f"Table extraction failed: {e}")
        return []

def save_output(data: dict, path: str) -> None:
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved output to {path}")
    except Exception as e:
        logger.exception(f"Saving output failed: {e}")
        raise

def process_s3_pdf(bucket_name: str, object_key: str, output_file: str) -> bool:
    try:
        textract_response = call_textract_with_bytes(bucket_name, object_key)
        text = extract_text(textract_response)
        tables = extract_tables(textract_response)

        save_output({
            "text": text,
            "tables": tables
        }, output_file)

        logger.info("PDF processed successfully.")
        return True

    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        return False

if __name__ == "__main__":
    import os

    bucket = os.environ.get("BUCKET_NAME", "your-s3-bucket")
    key = os.environ.get("OBJECT_KEY", "path/to/your/document.pdf")
    output_path = os.environ.get("OUTPUT_PATH", "output.json")

    success = process_s3_pdf(bucket, key, output_path)
    if success:
        logger.info("✅ Extraction completed.")
    else:
        logger.error("❌ Extraction failed.")
