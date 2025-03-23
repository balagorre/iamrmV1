import boto3
import json
import os  # Import the 'os' module
from textractcaller.t_call import call_textract, Textract_Features
from textractprettyprinter.t_pretty_print import get_text_from_layout_json, get_tables_string, convert_table_to_list
import logging

# --- Configuration ---
BEDROCK_REGION = 'us-east-1'  # Your Bedrock region (if needed later)
OUTPUT_DIR = 'textract_output'  # Directory to store Textract JSON results

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def extract_text_and_tables_from_pdf(pdf_path, output_dir=OUTPUT_DIR):
    """
    Extracts text, tables, and layout information from a PDF using Textract.
    Saves the raw JSON response and a processed version (text + tables).

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory to save the Textract output.

    Returns:
        A dictionary containing:
          - 'text':  The extracted plain text.
          - 'tables': A list of tables (each table is a list of lists).
          - 'json_path': Path to the saved raw Textract JSON.
          - 'processed_path': Path to the saved processed text and tables.
        Or None if an error occurs.
    """
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Call Textract with LAYOUT and TABLES features
        textract_json = call_textract(
            input_document=pdf_path,
            features=[Textract_Features.LAYOUT, Textract_Features.TABLES]
        )

        # --- Save Raw Textract JSON ---
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]  # Filename without extension
        json_path = os.path.join(output_dir, f"{base_filename}_textract.json")
        with open(json_path, 'w') as f:
            json.dump(textract_json, f, indent=4)
        logging.info(f"Saved raw Textract JSON to: {json_path}")


        # --- Processed Text and Tables ---
        text = get_text_from_layout_json(textract_json) # Get the plain text
        tables = []
        try:
            tables_string = get_tables_string(textract_json)
            table_list = convert_table_to_list(textract_json)
            if table_list: #check if tables are empty or not
               tables = table_list # Extract Tables
        except Exception as e:
            logging.warning(f"Error extracting tables, No Tables found: {e}")

        processed_path = os.path.join(output_dir, f"{base_filename}_processed.json")
        with open(processed_path, 'w') as f:
            json.dump({'text': text, 'tables': tables}, f, indent=4)
        logging.info(f"Saved processed text and tables to: {processed_path}")
        return {
            'text': text,
            'tables': tables,
            'json_path': json_path,
            'processed_path': processed_path
        }

    except Exception as e:
        logging.error(f"Error processing PDF {pdf_path}: {e}")
        return None

def load_processed_textract_output(processed_path):
    """Loads processed Textract output (text and tables) from a JSON file."""
    try:
        with open(processed_path, 'r') as f:
            data = json.load(f)
            return data['text'], data['tables']
    except Exception as e:
        logging.error(f"Error loading processed Textract output from {processed_path}: {e}")
        return None, None
# --- Main Execution (Example) ---

if __name__ == "__main__":
    whitepaper_path = 'path/to/whitepaper.pdf'  # Replace
    test_plan_path = 'path/to/test_plan.pdf'  # Replace
    test_results_path = 'path/to/test_results.pdf'  # Replace
    # Process Whitepaper
    whitepaper_data = extract_text_and_tables_from_pdf(whitepaper_path)
    if whitepaper_data:
        print(f"Whitepaper processing successful. Saved to: {whitepaper_data['processed_path']}")
        # Example of loading the processed data:
        # loaded_text, loaded_tables = load_processed_textract_output(whitepaper_data['processed_path'])

    # Process Test Plan
    test_plan_data = extract_text_and_tables_from_pdf(test_plan_path)
    if test_plan_data:
        print(f"Test Plan processing successful. Saved to: {test_plan_data['processed_path']}")

    # Process Test Results
    test_results_data = extract_text_and_tables_from_pdf(test_results_path)
    if test_results_data:
        print(f"Test Results processing successful. Saved to: {test_results_data['processed_path']}")
