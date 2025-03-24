import json
import logging
import os
import pandas as pd
from typing import List
from difflib import SequenceMatcher
from trp.trp2 import TDocument

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fuzzy_match(a: str, b: str, threshold: float = 0.7) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def extract_tables_with_trp_from_file(json_path: str, section_title: str, page_range: int = 2) -> List[pd.DataFrame]:
    """Loads Textract output JSON and extracts tables under the specified section."""
    with open(json_path, "r", encoding="utf-8") as f:
        response = json.load(f)

    tdoc = TDocument(response)
    section_page = None

    for page in tdoc.pages:
        for line in page.lines:
            if fuzzy_match(line.text, section_title):
                section_page = page.page
                logger.info(f"Found section '{section_title}' on page {section_page}")
                break
        if section_page:
            break

    if not section_page:
        logger.warning(f"Section '{section_title}' not found in the document.")
        return []

    allowed_pages = set(range(section_page, section_page + page_range + 1))
    dataframes = []

    for page in tdoc.pages:
        if page.page in allowed_pages:
            for table in page.tables:
                rows = []
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    rows.append(row_data)
                if rows:
                    df = pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) > 1 else pd.DataFrame(rows)
                    dataframes.append(df)

    logger.info(f"Found {len(dataframes)} table(s) under section '{section_title}'")
    return dataframes

def save_tables_to_files(dataframes: List[pd.DataFrame], output_dir: str, section_title: str):
    """Saves each DataFrame to a CSV and JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    for idx, df in enumerate(dataframes):
        base_name = f"{section_title.lower().replace(' ', '_')}_table_{idx + 1}"
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        json_path = os.path.join(output_dir, f"{base_name}.json")
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=4)
        logger.info(f"Saved table {idx + 1} to {csv_path} and {json_path}")

if __name__ == "__main__":
    json_file = "textract_output.json"            # Output from run_textract_job.py
    section = "Input Data"                         # Section to look for
    output_dir = "output_tables"                   # Folder to store results

    tables = extract_tables_with_trp_from_file(json_file, section)
    if tables:
        save_tables_to_files(tables, output_dir, section)

    for i, df in enumerate(tables):
        print(f"\n📄 Table {i+1}:\n", df)
