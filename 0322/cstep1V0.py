import boto3
import json
import re
import logging
from textractcaller.t_call import call_textract, Textract_Features
from textractprettyprinter.t_pretty_print import get_text_from_layout_json, convert_table_to_list

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def call_textract_with_bytes(bucket: str, key: str) -> dict:
    """Downloads the S3 PDF into memory and calls Textract with LAYOUT + TABLES."""
    try:
        s3 = boto3.client("s3")
        textract = boto3.client("textract")

        logger.info(f"Downloading s3://{bucket}/{key}")
        obj = s3.get_object(Bucket=bucket, Key=key)
        file_bytes = obj['Body'].read()

        logger.info("Calling Textract with LAYOUT and TABLES features (bytes mode)")
        response = call_textract(
            input_document=file_bytes,
            features=[Textract_Features.LAYOUT, Textract_Features.TABLES],
            boto3_textract_client=textract
        )

        if not response or 'Blocks' not in response:
            raise ValueError("Empty or invalid Textract response.")

        return response

    except Exception as e:
        logger.exception(f"Textract call failed: {e}")
        raise

def extract_text(textract_response: dict) -> str:
    """Extracts and cleans text from Textract response."""
    try:
        blocks = textract_response.get("Blocks", [])
        text = get_text_from_layout_json(
            textract_json={"Blocks": blocks},
            exclude_page_header=True,
            exclude_page_footer=True,
            exclude_page_number=True
        )
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        logger.exception("Failed to extract text")
        return ""

def extract_tables(textract_response: dict) -> list:
    """Extracts tables, returns [] on failure."""
    try:
        tables = convert_table_to_list(textract_response)
        return tables if isinstance(tables, list) else []
    except Exception as e:
        logger.warning("Table extraction failed")
        return []

def save_output(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved output to {path}")

def process_s3_pdf(bucket: str, key: str, output_path: str):
    try:
        response = call_textract_with_bytes(bucket, key)
        text = extract_text(response)
        tables = extract_tables(response)

        save_output({
            "text": text,
            "tables": tables
        }, output_path)

        logger.info("✅ PDF processed successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return False

# ---- Runner ----
if __name__ == "__main__":
    import os

    bucket = os.getenv("BUCKET_NAME", "your-s3-bucket-name")
    key = os.getenv("OBJECT_KEY", "your/path/to.pdf")
    output_path = os.getenv("OUTPUT_PATH", "output.json")

    process_s3_pdf(bucket, key, output_path)
