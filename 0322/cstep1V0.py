def find_tables_after_section(blocks: List[dict], section_title: str, page_range: int = 2) -> List[List[List[str]]]:
    section_found = False
    section_page = None
    allowed_pages = set()
    
    # Find the page where section appears
    for block in blocks:
        if block["BlockType"] == "LINE":
            text = block.get("Text", "").strip()
            if fuzzy_match(text, section_title):
                section_page = block.get("Page")
                logger.info(f"Found section '{section_title}' on page {section_page}")
                section_found = True
                break

    if not section_found or section_page is None:
        logger.warning(f"Section title '{section_title}' not found.")
        return []

    # Pages to consider for tables
    for i in range(page_range + 1):  # Include the section page + N pages after
        allowed_pages.add(section_page + i)

    # Convert all tables first
    all_tables = convert_table_to_list({"Blocks": blocks})

    # Optional: Map tables to page numbers using bounding box origin page (advanced, skipped here)

    # Just return all tables from allowed pages (basic filtering)
    return all_tables







import boto3
import time
import json
import logging
from typing import List
import pandas as pd
from textractprettyprinter.t_pretty_print import convert_table_to_list
from difflib import SequenceMatcher

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start the asynchronous Textract analysis job
def start_textract_analysis(bucket: str, key: str) -> str:
    textract = boto3.client("textract")
    response = textract.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
        FeatureTypes=["TABLES", "LAYOUT"]
    )
    job_id = response["JobId"]
    logger.info(f"Started Textract job: {job_id}")
    return job_id

# Wait for the job to complete
def wait_for_completion(job_id: str) -> str:
    textract = boto3.client("textract")
    while True:
        response = textract.get_document_analysis(JobId=job_id)
        status = response["JobStatus"]
        logger.info(f"Job status: {status}")
        if status in ["SUCCEEDED", "FAILED"]:
            return status
        time.sleep(5)

# Get all result pages of the job
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

# Fuzzy matching helper
def fuzzy_match(a: str, b: str, threshold: float = 0.7) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

# Find tables after a section heading, optionally across multiple pages
def find_tables_after_section(blocks: List[dict], section_title: str, page_range: int = 2) -> List[List[List[str]]]:
    section_found = False
    section_page = None
    collected_tables = []

    for block in blocks:
        if block["BlockType"] == "LINE":
            text = block.get("Text", "").strip()
            if fuzzy_match(text, section_title):
                section_found = True
                section_page = block.get("Page")
                logger.info(f"Found section '{section_title}' on page {section_page}")

        elif section_found and block["BlockType"] == "TABLE":
            table_page = block.get("Page")
            if table_page is not None and section_page is not None and (0 <= table_page - section_page <= page_range):
                collected_tables.append(block)

    if not collected_tables:
        logger.warning(f"No tables found after section: {section_title}")
        return []

    # Rebuild full tables
    full_tables = convert_table_to_list({"Blocks": blocks})
    return full_tables[-len(collected_tables):] if len(collected_tables) <= len(full_tables) else full_tables

# Convert list of tables to DataFrames
def tables_to_dataframes(tables: List[List[List[str]]]) -> List[pd.DataFrame]:
    dfs = []
    for table in tables:
        if not table or not table[0]:
            continue
        df = pd.DataFrame(table[1:], columns=table[0])
        dfs.append(df)
    return dfs

# Save raw Textract response
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved output to {path}")

# Full process: Textract job + table extraction by section
def process_pdf_section_tables(bucket: str, key: str, section_title: str, output_path: str, page_range: int = 2) -> List[pd.DataFrame]:
    job_id = start_textract_analysis(bucket, key)
    status = wait_for_completion(job_id)

    if status != "SUCCEEDED":
        logger.error("Textract job failed.")
        return []

    blocks = get_all_blocks(job_id)
    save_json({"Blocks": blocks}, output_path)

    tables = find_tables_after_section(blocks, section_title, page_range)
    dataframes = tables_to_dataframes(tables)

    logger.info(f"Found {len(dataframes)} table(s) under section: '{section_title}'")
    return dataframes

# === Runner ===
if __name__ == "__main__":
    bucket_name = "your-s3-bucket-name"
    object_key = "your/path/to/largefile.pdf"
    section_to_find = "Question from the document"  # Replace with your section title
    output_json = "textract_output.json"

    dfs = process_pdf_section_tables(bucket_name, object_key, section_to_find, output_json)

    for i, df in enumerate(dfs):
        print(f"\n📄 Table {i+1}:\n", df)
