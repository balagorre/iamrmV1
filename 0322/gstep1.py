import json
import logging
import os
import boto3
from typing import List, Dict, Any
import argparse
import pandas as pd

# --- Configuration ---
BEDROCK_REGION = 'us-east-1'  # Or your Bedrock region
CLAUDE_MODEL_ID = "anthropic.claude-v2"  # Or your preferred Claude model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def invoke_claude(prompt: str, max_tokens: int = 2000) -> str:
    """Invokes Claude with a given prompt and handles retries."""
    client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    try:
        body = json.dumps({
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": max_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
        })
        response = client.invoke_model(
            modelId=CLAUDE_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("completion")
    except Exception as e:
        logger.exception(f"Error invoking Claude: {e}")
        return ""

def extract_tables_with_claude(textract_json_path: str, chunk_size: int = 500) -> List[List[List[str]]]:
    """Extracts tables using Claude, with chunking."""
    try:
        with open(textract_json_path, 'r') as f:
            textract_data = json.load(f)

        if not isinstance(textract_data, dict) or 'Blocks' not in textract_data:
            logger.error(f"Invalid Textract JSON format: {textract_json_path}")
            return []

        blocks = textract_data['Blocks']
        all_tables = []

        for i in range(0, len(blocks), chunk_size):
            chunk = blocks[i:i + chunk_size]

            prompt = f"""You are an expert in extracting data from JSON responses of the AWS Textract service.
Here is a portion of a Textract JSON response. Extract all tables from this JSON.

{json.dumps(chunk, indent=2)}

The JSON contains a list of "Blocks".  Here's how to interpret them:

*   `"BlockType": "TABLE"`:  Indicates the start of a table.
*   `"BlockType": "CELL"`: Represents a cell within a table.
*   `"BlockType": "WORD"`: Represents a word.
*    `"BlockType": "PAGE"`: Represents a page.

Return the extracted tables as a JSON list of tables. Each table should be a list of rows, and each row should be a list of cell values (strings).
Do *not* include any explanations or extra text, *only* the JSON list.

Example Output (for a single table with 2 rows and 2 columns):
```json
[
  [
    ["Header 1", "Header 2"],
    ["Value 1", "Value 2"]
  ]
]
If No Tables, return an empty array '[]'.
JSON:
"""
logger.info(f"Sending prompt to Claude (chunk {i // chunk_size + 1})")
claude_response = invoke_claude(prompt, max_tokens=4096)
if not claude_response:
            logger.error(f"Claude returned an empty response for chunk {i // chunk_size + 1}.")
            continue

        try:
            chunk_tables = json.loads(claude_response)
            if isinstance(chunk_tables, list):
                all_tables.extend(chunk_tables)
            else:
                logger.error(f"Claude returned an unexpected type for chunk {i // chunk_size + 1}: {type(chunk_tables)}")
        except json.JSONDecodeError:
            logger.error(f"Claude response is not valid JSON for chunk {i // chunk_size + 1}: {claude_response}")

    return all_tables

except FileNotFoundError:
    logger.error(f"File not found: {textract_json_path}")
    return []
except Exception as e:
    logger.exception(f"Error extracting tables with Claude: {e}")
    return []
def save_tables_to_files(tables: List[List[List[str]]], output_dir: str, base_filename: str):
"""Saves extracted tables to CSV and JSON files."""
try:
os.makedirs(output_dir, exist_ok=True)
for i, table in enumerate(tables):
try:
df = pd.DataFrame(table[1:], columns=table[0]) if len(table) > 1 else pd.DataFrame(table)
csv_path = os.path.join(output_dir, f"{base_filename}table{i + 1}.csv")
json_path = os.path.join(output_dir, f"{base_filename}table{i + 1}.json")
df.to_csv(csv_path, index=False)
df.to_json(json_path, orient="records", indent=4)
logger.info(f"Saved table {i + 1} to {csv_path} and {json_path}")
except Exception as e:
logger.exception(f"Error saving table {i+1}: {e}")
except Exception as e:
logger.exception(f"Error creating output directory or saving files: {e}")

if name == "main":
parser = argparse.ArgumentParser(description="Extract tables from Textract JSON using Claude.")
parser.add_argument("json_file", help="Path to the Textract JSON file.")
parser.add_argument("-o", "--output_dir", default="extracted_tables", help="Output directory for tables.")
parser.add_argument("-c", "--chunk_size", type=int, default=500, help="Chunk size for processing Textract blocks.")
args = parser.parse_args()

base_filename = os.path.splitext(os.path.basename(args.json_file))[0]
base_filename = base_filename.replace("_textract", "")

tables = extract_tables_with_claude(args.json_file, chunk_size=args.chunk_size)
if tables:
    save_tables_to_files(tables, args.output_dir, base_filename)
    print(f"Extracted tables saved to: {args.output_dir}")
else:
    print("No tables extracted.")

for i, table in enumerate(tables):
    print(f"Extracted Table {i + 1}:")
    if isinstance(table, list):
        for row in table:
            print(row)
    else:
        print(f"Unexpected table format: {type(table)}")


















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

if __name__ == "__main__":
    bucket_name = "your-s3-bucket-name"  # REPLACE
    object_key_whitepaper = "path/to/your/whitepaper.pdf"  # REPLACE
    output_dir = "extracted_output"

    json_path = extractor.run_textract_job_async(bucket_name, object_key_whitepaper, output_dir)

    if json_path:
        # Use the Claude-based processor with chunking:
        # (No changes needed here - driver.py already uses the correct processor)
        tables = processor.extract_tables_with_claude(json_path) #, chunk_size=500) Removed chunk_size
        if tables:
            base_filename = os.path.splitext(os.path.basename(json_path))[0]
            base_filename = base_filename.replace("_textract","")
            processor.save_tables_to_files(tables, output_dir, base_filename)
            print(f"Extracted tables saved to: {output_dir}")
        else:
            print("No tables extracted by Claude.")

    else:
        logger.error("Textract job failed.")
