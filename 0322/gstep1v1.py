import boto3
import tempfile
import os
import traceback
import json
from botocore.exceptions import ClientError
from textractcaller.t_call import call_textract, Textract_Features
from textractprettyprinter.t_pretty_print import get_text_from_layout_json, get_tables_string, convert_table_to_list
import logging
from typing import List
from difflib import SequenceMatcher  # For string similarity

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_s3_file(bucket_name: str, object_key: str, local_file_path: str) -> bool:
    """Downloads a file from S3."""
    try:
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, object_key, local_file_path)
        logger.info(f"Downloaded s3://{bucket_name}/{object_key} to {local_file_path}")
        return True
    except ClientError as e:
        logger.error(f"Error downloading from S3: {e}")
        return False

def similar(a: str, b: str) -> float:
    """Calculates the similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()

def should_merge_tables(table1: List[List[str]], table2: List[List[str]]) -> bool:
    """
    Determines if two tables should be merged based on heuristics.
    """
    if not table1 or not table2:
        return False

    if len(table1[0]) != len(table2[0]):
        return False

    last_row_table1 = " ".join(table1[-1]).lower()
    first_row_table2 = " ".join(table2[0]).lower()

    if similar(last_row_table1, first_row_table2) > 0.6:
        return True

    return False

def merge_tables(tables: List[List[List[str]]]) -> List[List[List[str]]]:
    """Merges tables that span multiple pages."""
    merged_tables = []
    if not tables:
        return merged_tables

    current_table = tables[0]
    for next_table in tables[1:]:
        if should_merge_tables(current_table, next_table):
            current_table.extend(next_table[1:])
        else:
            merged_tables.append(current_table)
            current_table = next_table
    merged_tables.append(current_table)
    return merged_tables

def extract_text_and_tables_from_pdf(local_file_path: str) -> dict:
    """Extracts text and tables, excluding headers/footers/page numbers, and merges tables."""
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
            return {}

        print(f"Textract Response (truncated): {json.dumps(response)[:500]}")

        text = get_text_from_layout_json(response,
                                        exclude_header_footer=True,
                                        exclude_page_number=True)

        tables = []
        try:
            table_list = convert_table_to_list(response)
            if table_list:
                tables = table_list
        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")

        merged_tables = merge_tables(tables)
        return {'text': text, 'tables': merged_tables, 'response': response}

    except ClientError as e:
        logger.error(f"Textract ClientError: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error extracting text/tables: {e}")
        return {}


def save_processed_output(data: dict, output_file_path: str) -> None:
    """Saves the processed output to a JSON file."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4)
        logger.info(f"Saved processed output to {output_file_path}")
    except Exception as e:
        logger.error(f"Error saving output: {e}")
        traceback.print_exc()

def extract_from_s3_pdf(bucket_name: str, object_key: str, output_dir: str) -> bool:
    """Downloads PDF from S3, extracts text/tables, saves processed output."""
    local_file_path = None
    try:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(object_key))[0]
        output_file_path = os.path.join(output_dir, f"{base_filename}_processed.json")
        raw_json_path = os.path.join(output_dir, f"{base_filename}_textract.json") # Define raw_json_path

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            local_file_path = temp_file.name

            if not download_s3_file(bucket_name, object_key, local_file_path):
                return False

            extracted_data = extract_text_and_tables_from_pdf(local_file_path)

            if not extracted_data or not extracted_data.get('text', '').strip():
                logger.error("No text was extracted.")
                return False

            save_processed_output({'text': extracted_data['text'], 'tables': extracted_data['tables']}, output_file_path)

            with open(raw_json_path, 'w') as f: # Use raw_json_path here
                json.dump(extracted_data['response'], f, indent=4)
            logger.info(f"Saved raw Textract JSON to: {raw_json_path}")
            return True

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        traceback.print_exc()
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
