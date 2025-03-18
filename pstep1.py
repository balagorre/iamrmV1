import fitz  # PyMuPDF
import tabula
import pytesseract
from PIL import Image
import os
import json

def extract_text_from_pdf(pdf_path, output_dir):
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    Saves extracted text to a file.
    """
    os.makedirs(output_dir, exist_ok=True)
    extracted_text = ""
    
    # Open the PDF with PyMuPDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Processing {total_pages} pages from {pdf_path}...")
    
    for page_num in range(total_pages):
        page = doc[page_num]
        page_text = page.get_text()
        extracted_text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
        
        # Save individual page text for debugging/inspection
        with open(f"{output_dir}/page_{page_num + 1:03d}.txt", "w", encoding="utf-8") as f:
            f.write(page_text)
    
    # Save full extracted text
    full_text_path = os.path.join(output_dir, "extracted_text.txt")
    with open(full_text_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    print(f"Extracted text saved to {full_text_path}")
    return extracted_text

def extract_tables_from_pdf(pdf_path, output_dir):
    """
    Extracts tables from a PDF file using Tabula.
    Saves each table as a CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Extract tables from all pages
        tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True, lattice=True)
        
        if not tables:
            print("No tables found in the document.")
            return []
        
        table_files = []
        for i, table in enumerate(tables):
            if not table.empty:
                table_file = os.path.join(output_dir, f"table_{i + 1}.csv")
                table.to_csv(table_file, index=False)
                table_files.append(table_file)
                print(f"Table {i + 1} saved to {table_file}")
        
        return table_files
    except Exception as e:
        print(f"Error extracting tables: {e}")
        return []

def extract_images_from_pdf(pdf_path, output_dir):
    """
    Extracts images from a PDF file using PyMuPDF (fitz).
    Applies OCR to images if needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    image_files = []
    
    for page_num in range(total_pages):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image file
            image_file = os.path.join(output_dir, f"page_{page_num + 1}_image_{img_index + 1}.{image_ext}")
            with open(image_file, "wb") as f:
                f.write(image_bytes)
            
            image_files.append(image_file)
            
            # Apply OCR to the image if needed
            try:
                img = Image.open(image_file)
                ocr_text = pytesseract.image_to_string(img)
                
                if ocr_text.strip():
                    ocr_file = image_file.replace(f".{image_ext}", "_ocr.txt")
                    with open(ocr_file, "w", encoding="utf-8") as f:
                        f.write(ocr_text)
                    print(f"OCR applied to {image_file}, text saved to {ocr_file}")
            except Exception as e:
                print(f"Error applying OCR to {image_file}: {e}")
    
    print(f"Extracted {len(image_files)} images.")
    return image_files

def process_local_pdf(pdf_path, output_dir):
    """
    Full workflow to process a local PDF file.
    Extracts text, tables, and images.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\nStep 1: Extracting text...")
    extracted_text = extract_text_from_pdf(pdf_path, os.path.join(output_dir, "text"))

    print("\nStep 2: Extracting tables...")
    table_files = extract_tables_from_pdf(pdf_path, os.path.join(output_dir, "tables"))

    print("\nStep 3: Extracting images...")
    image_files = extract_images_from_pdf(pdf_path, os.path.join(output_dir, "images"))

    # Save metadata about extracted content
    metadata = {
        "pdf_path": pdf_path,
        "total_pages": len(fitz.open(pdf_path)),
        "text_file": os.path.join(output_dir, "text/extracted_text.txt"),
        "table_files": table_files,
        "image_files": image_files,
    }

    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nMetadata saved to {metadata_file}")
    
# Example usage
pdf_path = "./data/model_whitepaper.pdf"
output_dir = "./processed_content"

process_local_pdf(pdf_path, output_dir)


















import boto3
import time
import json
import os

def upload_pdf_to_s3(pdf_path, bucket_name, s3_key):
    """
    Uploads the PDF to an S3 bucket for processing by Textract.
    """
    s3 = boto3.client('s3')
    try:
        print(f"Uploading {pdf_path} to S3 bucket {bucket_name}...")
        s3.upload_file(pdf_path, bucket_name, s3_key)
        print(f"File uploaded successfully to {bucket_name}/{s3_key}.")
        return True
    except Exception as e:
        print(f"Error uploading file to S3: {str(e)}")
        return False

def start_textract_job(bucket_name, s3_key):
    """
    Starts a Textract job to analyze the PDF for text, tables, and forms.
    """
    textract = boto3.client('textract')
    try:
        print("Starting Textract job...")
        response = textract.start_document_analysis(
            DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': s3_key}},
            FeatureTypes=['TABLES', 'FORMS']  # Extract tables and forms
        )
        job_id = response['JobId']
        print(f"Textract Job ID: {job_id}")
        return job_id
    except Exception as e:
        print(f"Error starting Textract job: {str(e)}")
        return None

def wait_for_textract_job(job_id):
    """
    Waits for the Textract job to complete and returns the status.
    """
    textract = boto3.client('textract')
    while True:
        try:
            response = textract.get_document_analysis(JobId=job_id)
            status = response['JobStatus']
            if status in ['SUCCEEDED', 'FAILED']:
                print(f"Textract job completed with status: {status}")
                return status
            print(f"Job status: {status}. Waiting...")
            time.sleep(5)
        except Exception as e:
            print(f"Error checking Textract job status: {str(e)}")
            return None

def retrieve_textract_results(job_id):
    """
    Retrieves Textract results after the job is completed.
    Handles pagination for large documents.
    """
    textract = boto3.client('textract')
    results = []
    
    try:
        response = textract.get_document_analysis(JobId=job_id)
        results.append(response)
        
        next_token = response.get('NextToken')
        while next_token:
            response = textract.get_document_analysis(JobId=job_id, NextToken=next_token)
            results.append(response)
            next_token = response.get('NextToken')
        
        print("Textract results retrieved successfully.")
        return results
    except Exception as e:
        print(f"Error retrieving Textract results: {str(e)}")
        return None

def process_textract_results(results, output_dir):
    """
    Processes Textract results to extract text, tables, and forms.
    Saves extracted data into separate files for downstream analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize containers for extracted data
    extracted_text = ""
    tables = []
    
    # Process each result block
    for result in results:
        blocks = result['Blocks']
        
        for block in blocks:
            block_type = block['BlockType']
            
            if block_type == 'LINE':  # Extract text lines
                extracted_text += block['Text'] + "\n"
            
            elif block_type == 'TABLE':  # Extract table data
                table_data = []
                cells = [cell for cell in blocks if cell['BlockType'] == 'CELL']
                for cell in cells:
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    text = cell.get('Text', '')
                    table_data.append((row_index, col_index, text))
                
                # Convert table data into a structured format (CSV-like)
                max_row_index = max([cell[0] for cell in table_data])
                max_col_index = max([cell[1] for cell in table_data])
                
                table_matrix = [["" for _ in range(max_col_index)] for _ in range(max_row_index)]
                for row_idx, col_idx, text in table_data:
                    table_matrix[row_idx-1][col_idx-1] = text
                
                tables.append(table_matrix)
    
    # Save extracted text
    with open(os.path.join(output_dir, "extracted_text.txt"), "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    # Save extracted tables as CSV files
    for i, table in enumerate(tables):
        table_path = os.path.join(output_dir, f"table_{i+1}.csv")
        with open(table_path, "w", encoding="utf-8") as f:
            for row in table:
                f.write(",".join(row) + "\n")
    
    print("Extracted content saved successfully.")
    
# Main function to execute the enhanced step 1 workflow
def extract_content_from_pdf(pdf_path, bucket_name, s3_key, output_dir):
    """
    Executes the full workflow of uploading a PDF to S3,
    processing it with Textract, and saving extracted content locally.
    """
    # Step 1: Upload the PDF to S3
    if not upload_pdf_to_s3(pdf_path, bucket_name, s3_key):
        print("Failed to upload PDF to S3. Exiting.")
        return
    
    # Step 2: Start Textract job
    job_id = start_textract_job(bucket_name, s3_key)
    if not job_id:
        print("Failed to start Textract job. Exiting.")
        return
    
    # Step 3: Wait for Textract job completion
    status = wait_for_textract_job(job_id)
    if status != "SUCCEEDED":
        print("Textract job did not succeed. Exiting.")
        return
    
    # Step 4: Retrieve Textract results
    results = retrieve_textract_results(job_id)
    
    # Step 5: Process and save extracted content
    process_textract_results(results, output_dir)

# Example usage
pdf_path = "./data/model_whitepaper.pdf"
bucket_name = "your-s3-bucket-name"
s3_key = "documents/model_whitepaper.pdf"
output_dir = "./extracted_content"

extract_content_from_pdf(pdf_path, bucket_name, s3_key, output_dir)
