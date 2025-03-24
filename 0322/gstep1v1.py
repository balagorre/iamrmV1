import boto3
import tempfile
import os
import traceback
import json
from botocore.exceptions import ClientError
from textractcaller.t_call import call_textract, Textract_Features
from textractprettyprinter.t_pretty_print import get_text_from_layout_json, get_tables_string, convert_table_to_list
import logging
from typing import List, Dict  # Import Dict
from difflib import SequenceMatcher

# --- Configuration ---
SIMILARITY_THRESHOLD = 0.7  # Increased threshold slightly

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_s3_file(bucket_name: str, object_key: str, local_file_path: str) -> bool:
    """Downloads a file from S3, handling potential ClientErrors."""
    try:
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, object_key, local_file_path)
        logger.info(f"Downloaded s3://{bucket_name}/{object_key} to {local_file_path}")
        return True
    except ClientError as e:
        logger.error(f"Error downloading from S3: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error during S3 download: {e}")  # More specific logging
        return False

def similar(a: str, b: str) -> float:
    """Calculates the similarity ratio between two strings, handling empty strings."""
    if not a or not b:  # Handle empty strings gracefully
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() # used lower()


def should_merge_tables(table1: List[List[str]], table2: List[List[str]]) -> bool:
    """
    Determines if two tables should be merged, with improved robustness.
    """
    if not table1 or not table2:
        return False
    if not table1[0] or not table2[0]: # Added checks if not empty rows
        return False
    if len(table1[0]) != len(table2[0]):
        return False

    # Handle cases where tables might not have enough rows for comparison
    if len(table1) < 1 or len(table2) < 1:
        return False

    last_row_table1 = " ".join(table1[-1])
    first_row_table2 = " ".join(table2[0])

    return similar(last_row_table1, first_row_table2) >= SIMILARITY_THRESHOLD


def merge_tables(tables: List[List[List[str]]]) -> List[List[List[str]]]:
    """Merges tables with improved handling of edge cases."""
    if not tables:
        return []

    merged_tables: List[List[List[str]]] = []
    current_table = tables[0]

    for next_table in tables[1:]:
        if should_merge_tables(current_table, next_table):
            # Merge, skipping the header row of the *next* table *if* it exists
            if len(next_table) > 1:
                current_table.extend(next_table[1:])
            else:  # Handle case where next_table has only a header row
                current_table.extend(next_table)
        else:
            merged_tables.append(current_table)
            current_table = next_table

    merged_tables.append(current_table)  # Don't forget the last table
    return merged_tables
def extract_text_and_tables_from_pdf(local_file_path: str) -> Dict:
    """Extracts text/tables, handles errors, merges tables, and returns a dictionary."""
    try:
        logger.info(f"Extracting text from: {local_file_path}")

        with open(local_file_path, 'rb') as file:
            pdf_bytes = file.read()

        response = call_textract(
            input_document=pdf_bytes,
            features=[Textract_Features.LAYOUT, Textract_Features.TABLES]
        )

        if not response or 'Blocks' not in response:
            logger.error(f"Invalid Textract response: {response}")
            return {}  # Return empty dict on error

        print(f"Textract Response (truncated): {json.dumps(response)[:500]}")

        text = get_text_from_layout_json(response, exclude_header_footer=True, exclude_page_number=True)

        tables = []
        try:
            print("--- Attempting to extract tables ---")
            table_list = convert_table_to_list(response)
            print(f"Initial table_list: {table_list}")
            if table_list:
                tables = table_list
                print(f"Extracted {len(tables)} tables initially.")
            else:
                print("No tables extracted by convert_table_to_list.")
        except Exception as e:
            logger.exception(f"Error extracting tables: {e}")  # Use logger.exception
            print("--- Table extraction failed ---")

        merged_tables = merge_tables(tables)
        print(f"Merged tables: {merged_tables}")

        return {'text': text, 'tables': merged_tables, 'response': response}

    except ClientError as e:
        logger.exception(f"Textract ClientError: {e}")  # Use logger.exception
        return {}
    except Exception as e:
        logger.exception(f"Error extracting text/tables: {e}")  # Use logger.exception
        return {}

def save_processed_output(data: Dict, output_file_path: str) -> None:
    """Saves processed output to JSON, handling potential errors."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4)
        logger.info(f"Saved processed output to {output_file_path}")
    except Exception as e:
        logger.exception(f"Error saving output: {e}")  # Use logger.exception

def extract_from_s3_pdf(bucket_name: str, object_key: str, output_dir: str) -> bool:
    """Downloads PDF, extracts data, saves output, with robust error handling."""
    local_file_path = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(object_key))[0]
        output_file_path = os.path.join(output_dir, f"{base_filename}_processed.json")
        raw_json_path = os.path.join(output_dir, f"{base_filename}_textract.json")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            local_file_path = temp_file.name

            if not download_s3_file(bucket_name, object_key, local_file_path):
                return False

            extracted_data = extract_text_and_tables_from_pdf(local_file_path)

            if not extracted_data or 'text' not in extracted_data or not extracted_data['text'].strip():
                logger.error("No text was extracted.")
                return False
            # Save even if tables are empty
            save_processed_output({'text': extracted_data['text'], 'tables': extracted_data['tables']}, output_file_path)


            with open(raw_json_path, 'w') as f:
                json.dump(extracted_data['response'], f, indent=4)
            logger.info(f"Saved raw Textract JSON to: {raw_json_path}")
            return True

    except Exception as e:
        logger.exception(f"Error processing document: {e}")  # Use logger.exception
        return False

    finally:
        if local_file_path and os.path.exists(local_file_path):
            os.remove(local_file_path)
            logger.info(f"Deleted temp file: {local_file_path}")

if __name__ == "__main__":
    bucket_name = "your-s3-bucket-name"  # Replace
    object_key_whitepaper = "path/to/your/whitepaper.pdf"  # Replace
    output_dir = "extracted_output"

    success_whitepaper = extract_from_s3_pdf(bucket_name, object_key_whitepaper, output_dir)
    if success_whitepaper:
        logger.info(f"Whitepaper extraction successful!")
    else:
        logger.error("Whitepaper extraction failed.")
