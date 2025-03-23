import boto3
import json
import re
from textractcaller.t_call import call_textract, Textract_Features
from textractprettyprinter.t_pretty_print import get_text_from_layout_json
from anthropic import Anthropic
import concurrent.futures
import time
import logging

# --- Configuration ---
BEDROCK_REGION = 'us-east-1'
CLAUDE_MODEL_ID = "anthropic.claude-v2"
MAX_WORKERS = 5  # Adjust based on your Bedrock limits and desired concurrency
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1  # Seconds

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants (SECTION_PATTERNS - same as before) ---
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
# --- Helper Functions ---
def invoke_claude_with_retry(prompt, max_tokens=500):
    """Invokes Claude with retries and exponential backoff."""
    client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    retries = 0
    delay = INITIAL_RETRY_DELAY

    while retries < MAX_RETRIES:
        try:
            body = json.dumps({
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
            })
            response = client.invoke_model(
                modelId=CLAUDE_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body
            )
            response_body = json.loads(response.get("body").read())
            return response_body.get("completion")

        except client.exceptions.ThrottlingException:
            logging.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay)
            retries += 1
            delay *= 2  # Exponential backoff

        except Exception as e:
            logging.error(f"Error invoking Claude: {e}")
            return None  # Or raise the exception if you want to stop on other errors

    logging.error(f"Max retries reached. Failed to invoke Claude.")
    return None

# --- Step 2: PDF Extraction (No changes here) ---
def extract_text_from_pdf(pdf_path):
    """Extracts text and layout from a PDF using Textract."""
    try:
        textract_json = call_textract(
            input_document=pdf_path,
            features=[Textract_Features.LAYOUT, Textract_Features.TABLES]
        )
        return textract_json
    except Exception as e:
        print(f"Error in Textract processing: {e}")
        return None

def process_textract_response(textract_json):
    """Processes the raw Textract JSON response."""
    text = get_text_from_layout_json(textract_json)
    return text
# --- Step 3: Section Classification (Modified for Multi-threading) ---

def classify_section(text_chunk, context=None):
    """Classifies a text chunk using regex and Claude (no change in logic)."""
    text_chunk_lower = text_chunk.lower()
    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_chunk_lower):
                return section

    if context:
      prompt = f"""Given the following context:
      Previous Section Heading: {context.get('previous_heading', 'None')}
      Next Section Heading: {context.get('next_heading', 'None')}
      Text Chunk: {text_chunk}
      Classify this text chunk into one of the following categories: {', '.join(SECTION_PATTERNS.keys())}.  
      If it does not belong to any of these, return 'Other'.
      Category: """
    else:
       prompt = f"""Given the following text chunk:
        Text Chunk: {text_chunk}
        Classify this text chunk into one of the following categories: {', '.join(SECTION_PATTERNS.keys())}.
        If it does not belong to any of these, return 'Other'.
        Category: """

    return invoke_claude_with_retry(prompt) # Using the retry version



def classify_document_sections_multithreaded(document_text):
    """Classifies sections using multi-threading."""
    sections = {}
    current_section = "Other"
    sections[current_section] = []
    paragraphs = document_text.split("\n\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a list to hold futures and their corresponding paragraph indices
        futures = []
        for i, paragraph in enumerate(paragraphs):
            context = {}
            if i > 0:
                context['previous_heading'] = classify_section(paragraphs[i - 1])
            if i < len(paragraphs) - 1:
                context['next_heading'] = classify_section(paragraphs[i + 1])
            # Submit classification task to the executor
            future = executor.submit(classify_section, paragraph, context)
            futures.append((future, i)) # Store future with its index

        # Gather results as they become available
        for future, i in futures:
            predicted_section = future.result()  # This blocks until the result is available
            if predicted_section != "Other":
                current_section = predicted_section
                if current_section not in sections:
                    sections[current_section] = []
            if predicted_section == "Other":
                if current_section == "Other":
                    sections[current_section].append(paragraphs[i])
                else:
                    if current_section not in sections:
                        sections[current_section] = []
                    sections[current_section].append(paragraphs[i])
            else:
                sections[current_section].append(paragraphs[i])

    return sections

# --- Step 4: Model Information Extraction (Modified for Multi-threading) ---

def extract_model_info_multithreaded(whitepaper_sections):
    """Extracts model info using multi-threading."""
    model_info = {}
    prompts = {  # Same prompts as before
        'summary': "Provide a concise summary of the model's purpose, methodology, and key findings.",
        'inputs': "List the model's inputs, including data sources and variable names. Be as specific as possible.",
        'outputs': "List the model's outputs, including variable names and descriptions.",
        'calculations': "Describe the model's calculations, including any formulas or mathematical approaches used.",
        'assumptions': "List the key assumptions made in the development and implementation of this model.",
        'limitations': "List the limitations of this model.",
        'upstreams': "Describe the upstream data sources and systems that this model depends on.",
        'downstreams': "Describe the downstream systems or consumers that use the output of this model."
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for info_key, prompt in prompts.items():
            relevant_sections = []
            for section_name, section_text_chunks in whitepaper_sections.items():
                if info_key in SECTION_PATTERNS and section_name in SECTION_PATTERNS[info_key]:
                    relevant_sections.extend(section_text_chunks)
            combined_text = "\n\n".join(relevant_sections)
            full_prompt = f"{combined_text}\n\n{prompt}\n\nExtracted Information:"
            # Submit to executor and store the future
            futures[info_key] = executor.submit(invoke_claude_with_retry, full_prompt, 1000)

        # Gather results
        for info_key, future in futures.items():
            try:
                model_info[info_key] = future.result().strip()
                print(f"Claude response for extracting {info_key}: {model_info[info_key]}")
            except Exception as e:
                logging.error(f"Error extracting {info_key}: {e}")
                model_info[info_key] = None  # Or some other default value

    return model_info
# --- Step 5 and 6: Test Plan/Results Analysis (Similar Multi-threading) ---
# Apply the same multi-threading pattern as in Step 4 to these functions.
def analyze_test_plan_multithreaded(test_plan_sections):
    test_plan_info = {}
    prompts = {
        "test_cases": "List and describe each test case in the test plan, including its purpose and expected outcome.",
        "coverage": """Assess the test coverage. Does the test plan comprehensively cover the model's functionality, inputs, outputs,
        calculations, assumptions, and limitations as described in the whitepaper?  Identify any gaps or areas that are not adequately tested."""
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for key, prompt in prompts.items():
            relevant_sections = []
            for section_name, section_text_chunks in test_plan_sections.items():
                if key in SECTION_PATTERNS and section_name in SECTION_PATTERNS[key]:
                    relevant_sections.extend(section_text_chunks)
            combined_text = "\n\n".join(relevant_sections)
            full_prompt = f"{combined_text}\n\n{prompt}\n\nExtracted Information:"
            futures[key] = executor.submit(invoke_claude_with_retry, full_prompt, 1000)
        for key, future in futures.items():
            try:
                test_plan_info[key] = future.result().strip()
                print(f"Claude response for extracting {key} from Test Plan: {test_plan_info[key]}")
            except Exception as e:
                logging.error(f"Error extracting {key}: {e}")
                test_plan_info[key] = None
    return test_plan_info

def analyze_test_results_multithreaded(test_results_sections):
    test_results_info = {}
    prompts = {
        "results_summary": """Summarize the test results. For each test case, identify the actual outcome and whether it
                            passed or failed based on the expected outcome.""",
        "discrepancies": """Identify and explain any discrepancies between the expected outcomes (from the
                           test plan) and the actual outcomes observed in the test results."""
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for key, prompt in prompts.items():
            relevant_sections = []
            for section_name, section_text_chunks in test_results_sections.items():
                if key in SECTION_PATTERNS and section_name in SECTION_PATTERNS[key]:
                    relevant_sections.extend(section_text_chunks)

            combined_text = "\n\n".join(relevant_sections)
            full_prompt = f"{combined_text}\n\n{prompt}\n\nExtracted Information:"
            futures[key] = executor.submit(invoke_claude_with_retry, full_prompt, 1000)
        for key, future in futures.items():
            try:
                test_results_info[key] = future.result().strip()
                print(f"Claude response for extracting {key} from Test Results: {test_results_info[key]}")
            except Exception as e:
                logging.error(f"Error extracting {key}: {e}")
                test_results_info[key] = None
    return test_results_info

# --- Step 7: Reconciliation (Potentially Multi-threaded) ---

def reconcile_multithreaded(model_info, test_plan_info, test_results_info):
    """Reconciles information with multi-threading (if applicable)."""
    reconciliation_report = {}

    # --- Whitepaper vs. Test Plan ---
    prompt1 = f"""
    Model Information (from Whitepaper):
    {model_info}

    Test Plan Information:
    {test_plan_info}

    Based on the provided information, identify any gaps in test coverage.  Specifically:
    1. Are all model inputs, outputs, calculations, assumptions, and limitations mentioned in the whitepaper covered by test cases in the test plan?  List any missing items.
    2. Are there any areas of the model's functionality or behavior that are not adequately addressed by the test plan? Describe these gaps.
    """

    # --- Test Plan vs. Test Results ---
    prompt2 = f"""
        Test Plan Information:
        {test_plan_info}

        Test Results Information:
        {test_results_info}

        Based on the provided information, identify any discrepancies between the test plan and the test results. Specifically:
        1. Are there any test cases in the test plan that are missing corresponding results in the test results? List them.
        2. For each test case, compare the expected outcome (from the test plan) with the actual outcome (from the test results).  Identify and explain any differences.
        3. Report Test case Pass/Fail Status.
        """

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future1 = executor.submit(invoke_claude_with_retry, prompt1, 1500)
        future2 = executor.submit(invoke_claude_with_retry, prompt2, 1500)

        try:
            reconciliation_report['whitepaper_vs_test_plan'] = future1.result().strip()
            print(f"Claude response for reconciling whitepaper vs test plan: {reconciliation_report['whitepaper_vs_test_plan']}")
        except Exception as e:
            logging.error(f"Error in whitepaper vs. test plan reconciliation: {e}")
            reconciliation_report['whitepaper_vs_test_plan'] = "Error during reconciliation."

        try:
            reconciliation_report['test_plan_vs_test_results'] = future2.result().strip()
            print(f"Claude response for reconciling test plan vs test results: {reconciliation_report['test_plan_vs_test_results']}")
        except Exception as e:
            logging.error(f"Error in test plan vs. test results reconciliation: {e}")
            reconciliation_report['test_plan_vs_test_results'] = "Error during reconciliation."


    return reconciliation_report

# --- Step 8: Reporting ---

def generate_report(model_info, test_plan_info, test_results_info, reconciliation_report):
    """Generates a final report summarizing the findings."""

    report = f"""
    Model Audit Report
    ==================

    1. Model Summary (from Whitepaper):
    -----------------------------------
    {model_info.get('summary', 'No summary available.')}

    2. Extracted Model Information:
    -------------------------------
    Inputs: {model_info.get('inputs', 'N/A')}
    Outputs: {model_info.get('outputs', 'N/A')}
    Calculations: {model_info.get('calculations', 'N/A')}
    Assumptions: {model_info.get('assumptions', 'N/A')}
    Limitations: {model_info.get('limitations', 'N/A')}
    Upstreams: {model_info.get('upstreams', 'N/A')}
    Downstreams: {model_info.get('downstreams', 'N/A')}

    3. Test Plan Analysis:
    -----------------------
    Test Cases: {test_plan_info.get('test_cases', 'N/A')}
    Coverage Assessment: {test_plan_info.get('coverage', 'N/A')}

    4. Test Results Analysis:
    --------------------------
    Results Summary: {test_results_info.get('results_summary', 'N/A')}
    Discrepancies: {test_results_info.get('discrepancies', 'N/A')}

    5. Reconciliation Results:
    ---------------------------
    Whitepaper vs. Test Plan Gaps:
    {reconciliation_report.get('whitepaper_vs_test_plan', 'N/A')}

    Test Plan vs. Test Results Discrepancies:
    {reconciliation_report.get('test_plan_vs_test_results', 'N/A')}
    """

    print(report)
    #  Optionally save the report to a file:
    with open("model_audit_report.txt", "w") as f:
        f.write(report)


# --- Main Execution (Putting it all together) ---
def main():
    """Main function to orchestrate the entire process."""
    whitepaper_path = 'path/to/whitepaper.pdf'
    test_plan_path = 'path/to/test_plan.pdf'
    test_results_path = 'path/to/test_results.pdf'

    # Step 2: Extract Text from PDFs
    whitepaper_json = extract_text_from_pdf(whitepaper_path)
    test_plan_json = extract_text_from_pdf(test_plan_path)
    test_results_json = extract_text_from_pdf(test_results_path)

    if not all([whitepaper_json, test_plan_json, test_results_json]):
        logging.error("Failed to extract text from one or more documents.")
        return

    whitepaper_text = process_textract_response(whitepaper_json)
    test_plan_text = process_textract_response(test_plan_json)
    test_results_text = process_textract_response(test_results_json)

    # Step 3: Classify Sections (Multi-threaded)
    whitepaper_sections = classify_document_sections_multithreaded(whitepaper_text)
    test_plan_sections = classify_document_sections_multithreaded(test_plan_text)
    test_results_sections = classify_document_sections_multithreaded(test_results_text)


    # Step 4: Extract Model Info (Multi-threaded)
    model_info = extract_model_info_multithreaded(whitepaper_sections)

    # Step 5: Analyze Test Plan (Multi-threaded)
    test_plan_info = analyze_test_plan_multithreaded(test_plan_sections)

    # Step 6: Analyze Test Results (Multi-threaded)
    test_results_info = analyze_test_results_multithreaded(test_results_sections)

    # Step 7: Reconcile (Multi-threaded)
    reconciliation_report = reconcile_multithreaded(model_info, test_plan_info, test_results_info)

    # Step 8: Generate Report
    generate_report(model_info, test_plan_info, test_results_info, reconciliation_report)

if __name__ == "__main__":
    main()
