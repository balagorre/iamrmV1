import boto3
import json
import time
import logging
from tqdm import tqdm

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_textract_analysis(bucket: str, key: str) -> str:
    textract = boto3.client("textract")
    response = textract.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
        FeatureTypes=["TABLES", "LAYOUT"]
    )
    return response["JobId"]

def wait_for_completion(job_id: str) -> str:
    textract = boto3.client("textract")
    pbar = tqdm(total=100, desc="Textract Job Progress", bar_format="{l_bar}{bar} [ time left: {remaining} ]")
    progress = 0

    while True:
        response = textract.get_document_analysis(JobId=job_id)
        status = response["JobStatus"]
        if status == "SUCCEEDED":
            pbar.n = 100
            pbar.close()
            return status
        elif status == "FAILED":
            pbar.close()
            return status
        else:
            time.sleep(5)
            if progress < 95:
                progress += 5
                pbar.n = progress
                pbar.refresh()

def get_all_blocks_and_metadata(job_id: str):
    textract = boto3.client("textract")
    blocks = []
    next_token = None
    metadata = {}

    while True:
        kwargs = {"JobId": job_id}
        if next_token:
            kwargs["NextToken"] = next_token
        response = textract.get_document_analysis(**kwargs)
        blocks.extend(response["Blocks"])
        metadata = response.get("DocumentMetadata", metadata)
        next_token = response.get("NextToken")
        if not next_token:
            break

    return {"Blocks": blocks, "DocumentMetadata": metadata}

def run_and_save_textract(bucket: str, key: str, output_path: str):
    job_id = start_textract_analysis(bucket, key)
    status = wait_for_completion(job_id)

    if status != "SUCCEEDED":
        logger.error("Textract job failed.")
        return

    full_response = get_all_blocks_and_metadata(job_id)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_response, f, indent=4)
    logger.info(f"✅ Saved Textract result to {output_path}")

if __name__ == "__main__":
    bucket_name = "your-s3-bucket-name"
    object_key = "path/to/your/document.pdf"
    output_file = "textract_output.json"

    run_and_save_textract(bucket_name, object_key, output_file)
