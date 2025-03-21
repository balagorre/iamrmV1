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

    # Extract Tables
    tables = []
    for page in t_document.pages:
        for table in page.tables:
            table_data = [[cell.text.strip() for cell in row.cells] for row in table.rows]
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
