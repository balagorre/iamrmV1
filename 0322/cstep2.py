import json
import logging
import os
import pandas as pd
from typing import List
from difflib import SequenceMatcher
from trp.trp2 import TDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fuzzy_match(a: str, b: str, threshold: float = 0.7) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def extract_tables_with_trp_from_file(json_path: str, section_title: str, page_range: int = 2) -> List[pd.DataFrame]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Fallback if DocumentMetadata is missing
    if "Blocks" in raw and "DocumentMetadata" not in raw:
        logger.warning("⚠️ DocumentMetadata missing — patching it using PAGE blocks.")
        page_count = max([b.get("Page", 0) for b in raw["Blocks"] if b.get("BlockType") == "PAGE"] or [1])
        response = {
            "Blocks": raw["Blocks"],
            "DocumentMetadata": {"Pages": page_count}
        }
    else:
        response = raw

    tdoc = TDocument(response)
    section_page = None

    for page in tdoc.pages:
        for line in page.lines:
            if fuzzy_match(line.text, section_title):
                section_page = page.page_number
                logger.info(f"Found section '{section_title}' on page {section_page}")
                break
        if section_page:
            break

    if not section_page:
        logger.warning(f"Section '{section_title}' not found.")
        return []

    allowed_pages = set(range(section_page, section_page + page_range + 1))
    dataframes = []

    for page in tdoc.pages:
        if page.page_number in allowed_pages:
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
    os.makedirs(output_dir, exist_ok=True)
    for idx, df in enumerate(dataframes):
        base_name = f"{section_title.lower().replace(' ', '_')}_table_{idx + 1}"
        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        json_path = os.path.join(output_dir, f"{base_name}.json")
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=4)
        logger.info(f"✅ Saved table {idx + 1} to {csv_path} and {json_path}")

if __name__ == "__main__":
    json_file = "textract_output.json"
    section = "Input Data"
    output_dir = "output_tables"

    tables = extract_tables_with_trp_from_file(json_file, section)
    if tables:
        save_tables_to_files(tables, output_dir, section)

    for i, df in enumerate(tables):
        print(f"\n📄 Table {i+1}:\n", df)
