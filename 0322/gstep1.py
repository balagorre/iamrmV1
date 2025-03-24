import boto3
import os
import json
import time
from botocore.exceptions import ClientError
import logging
from tqdm import tqdm

# --- Configuration ---
OUTPUT_DIR = 'extracted_output'
POLL_INTERVAL = 5  # Seconds to wait between polling Textract
MAX_WAIT_TIME = 1800 # Seconds (30 minutes) - Maximum time to wait for Textract

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_s3_file(bucket_name: str, object_key: str, local_file_path: str) -> bool:
    """Downloads a file from S3 (no changes)."""
    try:
        s3_client = boto3.client('s3')
        s3_client.download_file(bucket_name, object_key, local_file_path)
        logger.info(f"Downloaded s3://{bucket_name}/{object_key} to {local_file_path}")
        return True
    except ClientError as e:
        logger.error(f"Error downloading from S3: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error during S3 download: {e}")
        return False

def start_textract_job(bucket_name: str, object_key: str, features: list) -> str | None:
    """Starts an asynchronous Textract job."""
    try:
        textract_client = boto3.client('textract')
        response = textract_client.start_document_analysis(
            DocumentLocation={
                'S3Object': {
                    'Bucket': bucket_name,
                    'Name': object_key
                }
            },
            FeatureTypes=features
        )
        return response['JobId']
    except ClientError as e:
        logger.exception(f"Error starting Textract job: {e}")
        return None

def get_textract_results(job_id: str) -> List[Dict]:
    """Retrieves results from a Textract job, handling pagination."""
    try:
        textract_client = boto3.client('textract')
        response = textract_client.get_document_analysis(JobId=job_id)
        results = [response]

        # Handle pagination
        while 'NextToken' in response:
            next_token = response['NextToken']
            response = textract_client.get_document_analysis(JobId=job_id, NextToken=next_token)
            results.append(response)
        return results
    except ClientError as e:
        logger.exception(f"Error getting Textract results: {e}")
        return []
    except Exception as e:
         logger.exception(f"Error getting Textract results: {e}")
         return []

def wait_for_textract_job(job_id: str) -> str:
    """Waits for a Textract job to complete, with polling and timeout."""
    start_time = time.time()
    with tqdm(total=None, desc=f"Waiting for Textract job {job_id}", unit="s", disable=False) as pbar:
        while True:
            results = get_textract_results(job_id)
            if not results:  # Check if result is valid
                return "FAILED"
            status = results[0]['JobStatus']  # Status is in the first response
            pbar.set_postfix({"status": status})  # Update status in progress bar

            if status in ('SUCCEEDED', 'FAILED', 'PARTIAL_SUCCESS'):
                return status

            if time.time() - start_time > MAX_WAIT_TIME:
                logger.error(f"Textract job {job_id} timed out after {MAX_WAIT_TIME} seconds.")
                return "TIMED_OUT"

            time.sleep(POLL_INTERVAL)
            pbar.update(POLL_INTERVAL)  # Increment progress bar by poll interval


def run_textract_job_async(bucket_name: str, object_key: str, output_dir: str = OUTPUT_DIR) -> str | None:
    """
    Downloads a PDF, runs Textract *asynchronously*, saves the raw JSON.
    Returns path to JSON file, or None on failure.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(object_key))[0]
        json_path = os.path.join(output_dir, f"{base_filename}_textract.json")
        temp_pdf_path = os.path.join(output_dir,f"{base_filename}.pdf")

        if not download_s3_file(bucket_name, object_key, temp_pdf_path):
            return None

        # Start the asynchronous Textract job
        job_id = start_textract_job(bucket_name, object_key, ["LAYOUT", "TABLES"])  # Use object key
        if not job_id:
            return None

        logger.info(f"Started Textract job: {job_id}")

        # Wait for the job to complete
        status = wait_for_textract_job(job_id)
        if status != "SUCCEEDED":
            logger.error(f"Textract job failed with status: {status}")
            return None

        # Get the results (handling pagination)
        results = get_textract_results(job_id)
        if not results:
             return None

        # Combine paginated results into a single JSON structure
        combined_response = {
            'DocumentMetadata': results[0]['DocumentMetadata'], #Keep first page metadata
            'Blocks': []
             }
        for result_page in results:
            combined_response['Blocks'].extend(result_page['Blocks'])


        # Save the combined raw JSON
        with open(json_path, 'w') as f:
            json.dump(combined_response, f, indent=4)
        logger.info(f"Saved raw Textract JSON to: {json_path}")

        return json_path

    except Exception as e:
        logger.exception(f"Error in run_textract_job_async: {e}")
        return None
    finally: # Clean up pdf
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            logger.info(f"Deleted temp file: {temp_pdf_path}")



import json
import logging
from typing import List, Dict
from difflib import SequenceMatcher
from textractprettyprinter.t_pretty_print import get_text_from_layout_json, convert_table_to_list
import os

# --- Configuration ---
SIMILARITY_THRESHOLD = 0.7

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def similar(a: str, b: str) -> float:
    """Calculates the similarity ratio between two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def should_merge_tables(table1: List[List[str]], table2: List[List[str]]) -> bool:
    """Determines if two tables should be merged."""
    if not isinstance(table1, list) or not isinstance(table2, list):
        return False
    if not table1 or not table2:
        return False
    if not table1[0] or not table2[0]:
        return False
    if len(table1[0]) != len(table2[0]):
        return False
    if len(table1) < 1 or len(table2) < 1:
        return False

    last_row_table1 = " ".join(table1[-1])
    first_row_table2 = " ".join(table2[0])

    return similar(last_row_table1, first_row_table2) >= SIMILARITY_THRESHOLD

def merge_tables(tables: List[List[List[str]]]) -> List[List[List[str]]]:
    """Merges tables."""
    if not isinstance(tables, list):
        logger.warning(f"merge_tables received unexpected input type: {type(tables)}. Returning empty list.")
        return []

    merged_tables: List[List[List[str]]] = []
    if not tables:
        return merged_tables

    current_table = tables[0]
    for next_table in tables[1:]:
        if should_merge_tables(current_table, next_table):
             if isinstance(current_table, list) and isinstance(next_table, list):
                if len(next_table) > 1:
                    current_table.extend(next_table[1:])
                else:
                    current_table.extend(next_table)
        else:
            if isinstance(current_table,list):
                merged_tables.append(current_table)
            current_table = next_table
    if isinstance(current_table,list):
        merged_tables.append(current_table)
    return merged_tables

def process_textract_output(json_path: str, output_dir: str) -> bool:
    """Loads Textract JSON, extracts text/tables, merges tables, saves output."""
    try:
        if not os.path.exists(json_path):
            logger.error(f"Input JSON file not found: {json_path}")
            return False

        with open(json_path, 'r') as f:
            response = json.load(f)

        if not response or 'Blocks' not in response:
            logger.error(f"Invalid Textract response in file: {json_path}")
            return False

        text = get_text_from_layout_json(response, exclude_header_footer=True, exclude_page_number=True)

        tables = []
        try:
            table_list = convert_table_to_list(response)
            if isinstance(table_list, list):
                tables = table_list
            else:
                logger.warning("Could not extract tables in expected format.")
        except Exception as e:
            logger.exception(f"Error extracting tables: {e}")

        merged_tables = merge_tables(tables)

        base_filename = os.path.splitext(os.path.basename(json_path))[0]
        base_filename = base_filename.replace("_textract", "")
        output_file_path = os.path.join(output_dir, f"{base_filename}_processed.json")

        save_processed_output({'text': text, 'tables': merged_tables}, output_file_path)
        return True

    except Exception as e:
        logger.exception(f"Error processing Textract output: {e}")
        return False

def save_processed_output(data: Dict, output_file_path: str) -> None:
    """Saves processed output to JSON."""
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4)
        logger.info(f"Saved processed output to {output_file_path}")
    except Exception as e:
        logger.exception(f"Error saving output: {e}")

if __name__ == "__main__":
    # Example usage (assuming you've already run textract_extractor_async.py)
    json_file_path = "extracted_output/whitepaper_textract.json"  # Replace
    output_directory = "extracted_output"
    success = process_textract_output(json_file_path, output_directory)

    if success:
        logger.info("Textract output processed successfully!")
    else:
        logger.error("Error processing Textract output.")


# driver.py
import textract_extractor_async as extractor
import textract_processor as processor
import logging

# Logging setup (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    bucket_name = "your-s3-bucket-name"  # REPLACE with your bucket name
    object_key_whitepaper = "path/to/your/whitepaper.pdf"  # REPLACE with your object key
    output_dir = "extracted_output"

    # Step 1: Run Textract Asynchronously
    json_path = extractor.run_textract_job_async(bucket_name, object_key_whitepaper, output_dir)

    if json_path:
        # Step 2: Process the Textract Output
        success = processor.process_textract_output(json_path, output_dir)
        if success:
            logger.info("Whitepaper processing completed successfully!")
        else:
            logger.error("Error processing Textract output.")
    else:
        logger.error("Textract job failed.")
