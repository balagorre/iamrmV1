import boto3
import json
import logging
from datetime import datetime
from textractcaller.t_call import call_textract, Textract_Features
from trp.trp2 import TDocumentSchema
from tqdm import tqdm

# Initialize logging
logging.basicConfig(level=logging.INFO)

s3_client = boto3.client('s3')

SECTION_PATTERNS = {
    'upstreams': ['upstream systems', 'data ingestion', 'data source architecture'],
    'downstreams': ['downstream systems', 'model consumers', 'system integration'],
    'model_vetting': ['model vetting', 'user vetting', 'approval workflow'],
    'model_summary': ['executive summary', 'model summary', 'introduction and scope', 'overview'],
    'inputs': ['model inputs', 'input variables', 'data sources', 'data dictionary', 'input features'],
    'outputs': ['model outputs', 'output variables', 'results', 'scorecard outputs', 'predictions'],
    'calculations': ['calculations', 'estimation method', 'model methodology', 'formulas', 'mathematical approach'],
    'monitoring': ['model monitoring', 'performance monitoring', 'ongoing monitoring'],
    'validation': ['model validation', 'backtesting', 'performance testing', 'benchmarking'],
    'assumptions': ['assumptions', 'adjustments', 'business assumptions'],
    'limitations': ['limitations', 'constraints', 'challenges', 'known issues'],
    'governance': ['governance', 'oversight', 'model governance'],
    'controls': ['controls', 'checks', 'internal controls', 'control framework'],
    'use_case': ['use case', 'application of model', 'business use case'],
    'risk_management': ['risk management', 'model risk', 'model risk management'],
    'development': ['model development', 'model design', 'development approach'],
    'implementation': ['model implementation', 'deployment details', 'production readiness'],
    'testing': ['testing strategy', 'test plan', 'unit testing', 'integration testing'],
    'reconciliation': ['reconciliation testing', 'test coverage', 'results reconciliation'],
    'research': ['future research', 'enhancements', 'research opportunities'],
    'references': ['references', 'bibliography', 'citations'],
    'business_context': ['business context', 'problem statement', 'business objective', 'model purpose'],
    'dependencies': ['system dependencies', 'model dependencies', 'library dependencies'],
    'retraining': ['model retraining', 'refresh frequency', 'update cycle'],
    'change_log': ['model changes', 'version history', 'change log', 'model adjustments'],
    'data_quality': ['data quality', 'missing values', 'data validation checks'],
    'integration': ['integration testing', 'deployment pipeline', 'integration plan'],
    'explainability': ['explainability', 'model interpretability', 'shap values'],
    'compliance': ['compliance checks', 'regulatory mapping', 'policy alignment'],
    'performance': ['performance metrics', 'kpis', 'model accuracy'],
    'alerts': ['threshold alerts', 'monitoring thresholds', 'early warning signals']
}

def classify_section(heading):
    heading_lower = heading.lower()
    for section_type, patterns in SECTION_PATTERNS.items():
        if any(p in heading_lower for p in patterns):
            return section_type
    return 'unclassified'

def is_toc_page(lines):
    indicators = ['table of contents', 'contents', '.....']
    return sum(1 for line in lines if any(ind in line.lower() for ind in indicators)) >= 2

def extract_text_tables(bucket, document_key, heading_height_threshold=0.03):
    logging.info("Starting Textract call for document: %s", document_key)
    textract_json = call_textract(
        input_document=f"s3://{bucket}/{document_key}",
        features=[Textract_Features.LAYOUT, Textract_Features.TABLES],
        boto3_textract_client=boto3.client('textract')
    )

    logging.info("Textract analysis complete. Parsing structured TRP document...")
    t_document = TDocumentSchema().load(textract_json)
    structured_pages = []
    last_known_section = 'unclassified'
    last_table = None

    logging.info("Beginning page-by-page parsing...")
    for page_index, page in tqdm(enumerate(t_document.pages), total=len(t_document.pages), desc="Processing Pages"):
        lines = [line.text.strip() for line in page.get_lines()]
        is_toc = is_toc_page(lines)

        page_content = {
            'page_number': page_index + 1,
            'headings': [],
            'paragraphs': [],
            'tables': [],
            'section_type': 'toc' if is_toc else last_known_section
        }

        if not is_toc:
            for line in page.get_lines():
                text = line.text.strip()
                if line.geometry.bounding_box.height > heading_height_threshold:
                    page_content['headings'].append(text)
                    section_guess = classify_section(text)
                    if section_guess != 'unclassified':
                        page_content['section_type'] = section_guess
                        last_known_section = section_guess
                else:
                    page_content['paragraphs'].append(text)

        for table_index, table in enumerate(page.tables):
            new_table_data = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            if last_table and last_table['section_type'] == page_content['section_type']:
                if last_table['data'] and new_table_data and len(last_table['data'][0]) == len(new_table_data[0]):
                    last_table['data'].extend(new_table_data)
                    continue

            table_data = {
                'table_number': table_index + 1,
                'section_type': page_content['section_type'],
                'data': new_table_data
            }
            page_content['tables'].append(table_data)
            last_table = table_data

        structured_pages.append(page_content)

    return {
        'model_metadata': {
            'document_key': document_key,
            'extracted_at': datetime.utcnow().isoformat()
        },
        'structured_pages': structured_pages
    }

def classify_table_by_header(headers):
    keywords = {
        "input_parameters": ['feature', 'variable', 'input'],
        "evaluation_metrics": ['accuracy', 'precision', 'recall', 'f1'],
        "confusion_matrix": ['actual', 'predicted', 'true positive', 'false positive'],
        "threshold_metrics": ['threshold', 'cutoff', 'probability']
    }
    for table_type, terms in keywords.items():
        if any(any(term in h.lower() for term in terms) for h in headers):
            return table_type
    return "general"

def classify_tables_in_output(structured_output):
    logging.info("Classifying tables by header keywords...")
    for page in structured_output.get("structured_pages", []):
        for table in page.get("tables", []):
            if table['data']:
                header_row = table['data'][0]
                table['table_type'] = classify_table_by_header(header_row)
    return structured_output

def export_output(output_data, output_bucket, output_prefix, local_filename):
    logging.info("Exporting structured output to S3 and local file...")
    s3_key = f"{output_prefix}/structured_output.json"
    s3_client.put_object(
        Bucket=output_bucket,
        Key=s3_key,
        Body=json.dumps(output_data, indent=2).encode('utf-8')
    )
    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    logging.info(f"Output exported to S3 (s3://{output_bucket}/{s3_key}) and local file ({local_filename})")

def run_full_extraction_pipeline(input_bucket, input_pdf_key, output_bucket, output_prefix, local_output_path):
    logging.info("Starting full extraction pipeline for: %s", input_pdf_key)
    structured_output = extract_text_tables(bucket=input_bucket, document_key=input_pdf_key)
    structured_output = classify_tables_in_output(structured_output)
    full_output = {
        "model_metadata": structured_output["model_metadata"],
        "structured_pages": structured_output["structured_pages"]
    }
    export_output(
        output_data=full_output,
        output_bucket=output_bucket,
        output_prefix=output_prefix,
        local_filename=local_output_path
    )
    logging.info("✅ Full pipeline completed successfully.")

if __name__ == "__main__":
    input_bucket = "your-input-s3-bucket"
    input_pdf_key = "documents/sample-model.pdf"
    output_bucket = "your-output-s3-bucket"
    output_prefix = "outputs/sample-model"
    local_output_path = "sample-model-output.json"

    run_full_extraction_pipeline(
        input_bucket=input_bucket,
        input_pdf_key=input_pdf_key,
        output_bucket=output_bucket,
        output_prefix=output_prefix,
        local_output_path=local_output_path
    )
