import json
import logging
import os
import pandas as pd
from typing import List, Dict
from difflib import SequenceMatcher
from trp.trp2 import TDocument
import argparse  # Import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fuzzy_match(a: str, b: str, threshold: float = 0.7) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def extract_tables_with_trp_from_file(json_path: str, section_title: str, page_range: int = 2) -> List[pd.DataFrame]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # --- Improved Patching Logic ---
        if "Blocks" in raw and "DocumentMetadata" not in raw:
            logger.warning("⚠️ DocumentMetadata missing — patching it and adding PAGE blocks.")

            # 1. Find existing PAGE blocks (if any)
            existing_page_blocks = [b for b in raw["Blocks"] if b.get("BlockType") == "PAGE"]
            page_count = len(existing_page_blocks)

            # 2. If no PAGE blocks, we need to create them
            if page_count == 0:
                logger.warning("    No PAGE blocks found.  Creating them.")
                # Find max page number from other blocks (LINE, WORD, etc.)
                max_page = 0
                for block in raw["Blocks"]:
                    if "Page" in block:
                        max_page = max(max_page, block["Page"])
                page_count = max_page if max_page > 0 else 1 # Ensure at least one page

                # Create PAGE blocks
                new_page_blocks = []
                for i in range(1, page_count + 1):
                    new_page_blocks.append({
                        "BlockType": "PAGE",
                        "Id": f"page-{i}",  # Create a unique ID
                        "Page": i,
                         "Relationships": [] # Initialize empty relationships
                    })

                # Add new PAGE blocks to the beginning of the Blocks list
                raw["Blocks"] = new_page_blocks + raw["Blocks"]

                #Update existing blocks with relationships to the new PAGE blocks
                page_block_map = {block["Page"]: block for block in new_page_blocks}
                for block in raw["Blocks"]:
                    if "Page" in block and block["Page"] in page_block_map:
                        page_block = page_block_map[block["Page"]]
                        if not any(r["Type"] == "CHILD" and block["Id"] in r["Ids"] for r in page_block.get("Relationships",[])):
                            if "Relationships" not in page_block:
                                page_block["Relationships"] = []
                            page_block["Relationships"].append({"Type": "CHILD", "Ids": [block["Id"]]})
            else:
                logger.info("    PAGE blocks already exist.")


            # 3. Add (or update) DocumentMetadata
            response = {
                "Blocks": raw["Blocks"],
                "DocumentMetadata": {"Pages": page_count}
            }
            logger.info(f"    Patched DocumentMetadata. Pages: {page_count}")

        else:
            response = raw
            if "Blocks" not in response:
               logger.error(f"Invalid Textract response in file: {json_path}") # added validation
               return []


        # --- Rest of the function remains the same ---
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

    except FileNotFoundError:
        logger.error(f"File not found: {json_path}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in file: {json_path}")
        return []
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return []



def save_tables_to_files(dataframes: List[pd.DataFrame], output_dir: str, section_title: str):
    try:
        os.makedirs(output_dir, exist_ok=True)
        for idx, df in enumerate(dataframes):
            base_name = f"{section_title.lower().replace(' ', '_')}_table_{idx + 1}"
            csv_path = os.path.join(output_dir, f"{base_name}.csv")
            json_path = os.path.join(output_dir, f"{base_name}.json")
            try:
                df.to_csv(csv_path, index=False)
                df.to_json(json_path, orient="records", indent=4)
                logger.info(f"✅ Saved table {idx + 1} to {csv_path} and {json_path}")
            except Exception as e:
                logger.exception(f"Error saving table {idx + 1}: {e}")
    except Exception as e:
        logger.exception(f"Error creating output directory: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract tables from Textract JSON.")
    parser.add_argument("json_file", help="Path to the Textract JSON file.")
    parser.add_argument("section", help="Section title to search for.")
    parser.add_argument("-o", "--output_dir", default="output_tables", help="Output directory for tables.")
    parser.add_argument("-p", "--page_range", type=int, default=2, help="Number of pages to search after section title.")
    args = parser.parse_args()

    tables = extract_tables_with_trp_from_file(args.json_file, args.section, args.page_range)
    if tables:
        save_tables_to_files(tables, args.output_dir, args.section)

        for i, df in enumerate(tables):
            print(f"\n📄 Table {i+1}:\n", df)
    else:
        print("No Tables Found")
