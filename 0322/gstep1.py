import boto3
import os
import json
import time
from botocore.exceptions import ClientError
from textractcaller.t_call import call_textract, Textract_Features
import logging
from tqdm import tqdm
from typing import List, Dict

# --- Configuration ---
OUTPUT_DIR = 'extracted_output'
POLL_INTERVAL = 5  # Seconds to wait between polling Textract
MAX_WAIT_TIME = 1800 # Seconds (30 minutes)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
            if not results:
                return "FAILED"
            status = results[0]['JobStatus']
            pbar.set_postfix({"status": status})

            if status in ('SUCCEEDED', 'FAILED', 'PARTIAL_SUCCESS'):
                return status

            if time.time() - start_time > MAX_WAIT_TIME:
                logger.error(f"Textract job {job_id} timed out after {MAX_WAIT_TIME} seconds.")
                return "TIMED_OUT"

            time.sleep(POLL_INTERVAL)
            pbar.update(POLL_INTERVAL)


def run_textract_job_async(bucket_name: str, object_key: str, output_dir: str = OUTPUT_DIR) -> str | None:
    """Downloads PDF, runs Textract *asynchronously*, saves raw JSON. Returns JSON path."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(object_key))[0]
        json_path = os.path.join(output_dir, f"{base_filename}_textract.json")
        temp_pdf_path = os.path.join(output_dir,f"{base_filename}.pdf")

        if not download_s3_file(bucket_name, object_key, temp_pdf_path):
            return None

        job_id = start_textract_job(bucket_name, object_key, ["LAYOUT", "TABLES"])
        if not job_id:
            return None

        logger.info(f"Started Textract job: {job_id}")
        status = wait_for_textract_job(job_id)

        if status != "SUCCEEDED":
            logger.error(f"Textract job failed with status: {status}")
            return None

        results = get_textract_results(job_id)
        if results:
            combined_response = {
                'DocumentMetadata': results[0]['DocumentMetadata'],
                'Blocks': []
                }
            for result_page in results:
                combined_response['Blocks'].extend(result_page['Blocks'])

            with open(json_path, 'w') as f:
                json.dump(combined_response, f, indent=4)
            logger.info(f"Saved raw Textract JSON to: {json_path}")
            return json_path
        else:
           logger.error("Failed to get Textract results.")
           return None
    except Exception as e:
        logger.exception(f"Error in run_textract_job_async: {e}")
        return None
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
            logger.info(f"Deleted temp file: {temp_pdf_path}")



