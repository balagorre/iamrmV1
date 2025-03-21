import boto3
import fitz  # PyMuPDF
import json
import concurrent.futures
import logging
from datetime import datetime
from textractcaller.t_call import call_textract, Textract_Features
from textractprettyprinter.t_pretty_print import get_text_from_layout_json
from trp.trp2 import TDocumentSchema, TBlockType

# Initialize logging
logging.basicConfig(level=logging.INFO)

s3_client = boto3.client('s3')
rekognition_client = boto3.client('rekognition')

# Enhanced Step 1: Structured Text and Table Extraction with Dynamic Tagging
def extract_text_tables(bucket, document_key, heading_height_threshold=0.03):
    try:
        textract_json = call_textract(
            input_document=f"s3://{bucket}/{document_key}",
            features=[Textract_Features.LAYOUT, Textract_Features.TABLES],
            boto3_textract_client=boto3.client('textract')
        )

        t_document = TDocumentSchema().load(textract_json)
        structured_pages = []

        for page_index, page in enumerate(t_document.pages, start=1):
            page_content = {
                'page_number': page_index,
                'headings': [],
                'paragraphs': [],
                'tables': []
            }

            for block in page.blocks:
                if block.block_type == TBlockType.LINE:
                    if block.geometry.bounding_box.height > heading_height_threshold:
                        page_content['headings'].append(block.text.strip())
                    else:
                        page_content['paragraphs'].append(block.text.strip())

            for table_index, table in enumerate(page.tables, start=1):
                table_data = {
                    'table_number': table_index,
                    'data': [[cell.text.strip() for cell in row.cells] for row in table.rows]
                }
                page_content['tables'].append(table_data)

            structured_pages.append(page_content)

        logging.info("Text and tables extracted successfully.")
        return structured_pages

    except Exception as e:
        logging.error(f"Error extracting text and tables: {e}")
        raise

# Step 2: Extract images using PyMuPDF (optimized for large PDFs)
def extract_images(bucket, pdf_key, output_bucket, output_prefix):
    try:
        pdf_obj = s3_client.get_object(Bucket=bucket, Key=pdf_key)
        pdf_bytes = pdf_obj['Body'].read()

        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        extracted_images = []

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_name = f"{output_prefix}/page_{page_num + 1}_image_{img_index + 1}.{image_ext}"

                s3_client.put_object(Bucket=output_bucket, Key=image_name, Body=image_bytes)
                extracted_images.append(image_name)

        logging.info("Images extracted successfully.")
        return extracted_images

    except Exception as e:
        logging.error(f"Error extracting images: {e}")
        raise

# Helper function for parallel image analysis
def analyze_single_image(bucket, image_key):
    try:
        textract_response = call_textract(
            input_document=f"s3://{bucket}/{image_key}",
            features=[Textract_Features.LAYOUT],
            boto3_textract_client=boto3.client('textract')
        )
        image_text = get_text_from_layout_json(textract_response)

        rekognition_response = rekognition_client.detect_labels(
            Image={'S3Object': {'Bucket': bucket, 'Name': image_key}},
            MaxLabels=10
        )

        labels = [{'Label': label['Name'], 'Confidence': label['Confidence']}
                  for label in rekognition_response['Labels']]

        return image_key, {'text': image_text, 'labels': labels}

    except Exception as e:
        logging.error(f"Error analyzing image {image_key}: {e}")
        return image_key, {'error': str(e)}

# Step 3: Analyze images using parallel processing
def analyze_images(bucket, images):
    analysis_results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze_single_image, bucket, image) for image in images]
        for future in concurrent.futures.as_completed(futures):
            image_key, result = future.result()
            analysis_results[image_key] = result

    logging.info("Image analysis completed successfully.")
    return analysis_results

# Full integrated workflow with enhanced structured tagging, logging, and metadata
def full_document_processing(input_bucket, document_key, output_bucket, output_prefix):
    timestamp = datetime.utcnow().isoformat()

    structured_pages = extract_text_tables(input_bucket, document_key)
    images = extract_images(input_bucket, document_key, output_bucket, output_prefix)
    image_analysis = analyze_images(output_bucket, images)

    results = {
        'timestamp': timestamp,
        'input_document': document_key,
        'structured_pages': structured_pages,
        'images_analysis': image_analysis
    }

    s3_client.put_object(
        Bucket=output_bucket,
        Key=f"{output_prefix}/structured_document_analysis.json",
        Body=json.dumps(results).encode('utf-8')
    )

    logging.info("Complete structured extraction and analysis workflow executed successfully.")

# Example execution
# full_document_processing('your-input-bucket', 'path/to/document.pdf', 'your-output-bucket', 'processed_results')



































import boto3
import fitz  # PyMuPDF
import json
import time
from textractcaller.t_call import call_textract, Textract_Features
from textractprettyprinter.t_pretty_print import get_text_from_layout_json
from trp.trp2 import TDocumentSchema

s3_client = boto3.client('s3')
rekognition_client = boto3.client('rekognition')

# Step 1: Extract text and tables using Textract with pagination handling for large documents
def extract_text_tables(bucket, document_key):
    textract_json = call_textract(
        input_document=f"s3://{bucket}/{document_key}",
        features=[Textract_Features.LAYOUT, Textract_Features.TABLES],
        boto3_textract_client=boto3.client('textract')
    )

    # Structured Parsing
    t_document = TDocumentSchema().load(textract_json)

    # Extract Clean Text excluding irrelevant parts
    clean_text = get_text_from_layout_json(
        textract_json,
        exclude_figure_text=True,
        exclude_page_header=True,
        exclude_page_footer=True,
        exclude_page_number=True
    )

    # Extract Tables with metadata (page number and table index)
    tables = []
    for page_index, page in enumerate(t_document.pages, start=1):
        for table_index, table in enumerate(page.tables, start=1):
            table_data = {
                'page_number': page_index,
                'table_number': table_index,
                'data': [[cell.text.strip() for cell in row.cells] for row in table.rows]
            }
            tables.append(table_data)

    return clean_text, tables

# Step 2: Extract images using PyMuPDF (optimized for large PDFs)
def extract_images(bucket, pdf_key, output_bucket, output_prefix):
    pdf_obj = s3_client.get_object(Bucket=bucket, Key=pdf_key)
    pdf_bytes = pdf_obj['Body'].read()

    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    extracted_images = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_name = f"{output_prefix}/page_{page_num + 1}_image_{img_index + 1}.{image_ext}"

            s3_client.put_object(Bucket=output_bucket, Key=image_name, Body=image_bytes)
            extracted_images.append(image_name)

    return extracted_images

# Step 3: Analyze images using Textract (OCR) and Rekognition (Labels)
def analyze_images(bucket, images):
    analysis_results = {}

    for image_key in images:
        # OCR Text
        textract_response = call_textract(
            input_document=f"s3://{bucket}/{image_key}",
            features=[Textract_Features.LAYOUT],
            boto3_textract_client=boto3.client('textract')
        )
        image_text = get_text_from_layout_json(textract_response)

        # Image labels using Rekognition
        rekognition_response = rekognition_client.detect_labels(
            Image={'S3Object': {'Bucket': bucket, 'Name': image_key}},
            MaxLabels=10
        )

        labels = [{'Label': label['Name'], 'Confidence': label['Confidence']}
                  for label in rekognition_response['Labels']]

        analysis_results[image_key] = {'text': image_text, 'labels': labels}

    return analysis_results

# Full integrated workflow optimized for large documents
def full_document_processing(input_bucket, document_key, output_bucket, output_prefix):
    # Step 1: Extract text and tables
    text, tables = extract_text_tables(input_bucket, document_key)

    # Step 2: Extract images
    images = extract_images(input_bucket, document_key, output_bucket, output_prefix)

    # Step 3: Analyze images
    image_analysis = analyze_images(output_bucket, images)

    # Consolidate and store results
    results = {
        'text': text,
        'tables': tables,
        'images_analysis': image_analysis
    }

    s3_client.put_object(
        Bucket=output_bucket,
        Key=f"{output_prefix}/final_document_analysis.json",
        Body=json.dumps(results).encode('utf-8')
    )

    print("Complete extraction and analysis workflow executed successfully.")

# Example execution (replace placeholders with actual values)
# full_document_processing('your-input-bucket', 'path/to/document.pdf', 'your-output-bucket', 'processed_results')
