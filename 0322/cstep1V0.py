import boto3
import time
import json
import logging
from typing import List
import pandas as pd
from textractprettyprinter.t_pretty_print import convert_table_to_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Step 1: Start Textract Async Job ===
def start_textract_analysis(bucket: str, key: str) -> str:
    textract = boto3.client("textract")
    response = textract.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
        FeatureTypes=["TABLES", "LAYOUT"]
    )
    job_id = response["JobId"]
    logger.info(f"Started Textract job: {job_id}")
    return job_id

# === Step 2: Wait for Job Completion ===
def wait_for_completion(job_id: str) -> str:
    textract = boto3.client("textract")
    while True:
        response = textract.get_document_analysis(JobId=job_id)
        status = response["JobStatus"]
        logger.info(f"Job status: {status}")
        if status in ["SUCCEEDED", "FAILED"]:
            return status
        time.sleep(5)

# === Step 3: Get All Textract Blocks ===
def get_all_blocks(job_id: str) -> List[dict]:
    textract = boto3.client("textract")
    blocks = []
    next_token = None

    while True:
        kwargs = {"JobId": job_id}
        if next_token:
            kwargs["NextToken"] = next_token

        response = textract.get_document_analysis(**kwargs)
        blocks.extend(response["Blocks"])
        next_token = response.get("NextToken")

        if not next_token:
            break

    return blocks

# === Step 4: Find Tables After a Section Heading ===
def find_tables_after_section(blocks: List[dict], section_title: str) -> List[List[List[str]]]:
    section_found = False
    extracted_tables = []
    current_table = []

    for block in blocks:
        if block["BlockType"] == "LINE":
            text = block.get("Text", "").strip().lower()
            if section_title.lower() in text:
                section_found = True

        elif block["BlockType"] == "TABLE" and section_found:
            # Optional: You could filter by page here too
            current_table.append(block)

    if not current_table:
        logger.warning(f"No tables found after section: {section_title}")
        return []

    # Convert using textractprettyprinter (entire document)
    full_tables = convert_table_to_list({"Blocks": blocks})

    # Return all tables after section
    return full_tables[len(full_tables) - len(current_table):]

# === Step 5: Convert to DataFrames ===
def tables_to_dataframes(tables: List[List[List[str]]]) -> List[pd.DataFrame]:
    dfs = []
    for table in tables:
        if not table or not table[0]: continue  # Skip empty
        df = pd.DataFrame(table[1:], columns=table[0])
        dfs.append(df)
    return dfs

# === Step 6: Save to JSON for review ===
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved output to {path}")

# === Main Entry Point ===
def process_pdf_section_tables(bucket: str, key: str, section_title: str, output_path: str) -> List[pd.DataFrame]:
    job_id = start_textract_analysis(bucket, key)
    status = wait_for_completion(job_id)

    if status != "SUCCEEDED":
        logger.error("Textract job failed.")
        return []

    blocks = get_all_blocks(job_id)

    # Optional: Save full response
    save_json({"Blocks": blocks}, output_path)

    tables = find_tables_after_section(blocks, section_title)
    dataframes = tables_to_dataframes(tables)

    logger.info(f"Found {len(dataframes)} table(s) under section: '{section_title}'")
    return dataframes

# === Example Runner ===
if __name__ == "__main__":
    bucket_name = "your-s3-bucket-name"
    object_key = "path/to/your/largefile.pdf"
    section_to_find = "Input Data"
    output_json = "textract_output.json"

    dfs = process_pdf_section_tables(bucket_name, object_key, section_to_find, output_json)

    for i, df in enumerate(dfs):
        print(f"\n📄 Table {i+1}:\n", df)
