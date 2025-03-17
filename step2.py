# Inside Step 2's code:

from jsonschema import validate
from jsonschema.exceptions import ValidationError  # <--- Add this import


def process_file_chunk(file_path, artifact_type):
    """Processes a single chunk of a preprocessed file."""
    try:
        with open(file_path, 'r') as f:
            preprocessed_data = json.load(f)

        extracted_data_text = None  # Initialize here

        if artifact_type == 'whitepaper':
            prompt_template = get_whitepaper_prompt_template()
            few_shot_example = get_whitepaper_few_shot_example()
            schema = get_whitepaper_output_schema()
        elif artifact_type == 'testplan':
            prompt_template = get_testplan_prompt_template()
            few_shot_example = get_testplan_few_shot_example()
            schema = get_testplan_output_schema()
        elif artifact_type == 'testresults':
            prompt_template = get_testresults_prompt_template()
            few_shot_example = get_testresults_few_shot_example()
            schema = get_testresults_output_schema()
        elif artifact_type == 'reconciliationreport':
            prompt_template = get_reconciliation_report_prompt_template()
            few_shot_example = get_reconciliation_report_few_shot_example()
            schema = get_reconciliation_report_output_schema()
        else:
            raise ValueError(f"Invalid artifact type: {artifact_type}")

        prompt = prompt_template.format(preprocessed_text=preprocessed_data['text'], few_shot_example=few_shot_example)

        response = bedrock.invoke_model(
            body=json.dumps({
                "prompt": f"\n\nHuman:{prompt}\n\nAssistant:",
                "max_tokens_to_sample": 4096,
                "temperature": 0.1,
                "top_p": 0.9,
            }),
            modelId="anthropic.claude-v3-sonnet",
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response.get('body').read())
        extracted_data_text = response_body.get('completion')  # Now assigned within the conditional

        if extracted_data_text is None:
            print("Error: Claude 3 returned an empty completion.")
            return None

        try:
            extracted_data = json.loads(extracted_data_text)
        except json.JSONDecodeError as e:
            print(f"Error: JSONDecodeError: {e}")
            print(f"Invalid JSON: {extracted_data_text}")
            return None

        try:
            validate(instance=extracted_data, schema=schema)
        except ValidationError as e:
            print(f"Error: JSON Schema Validation Failed: {e}")
            print(f"For data: {extracted_data}")  # Print the data that failed validation
            return None

        return extracted_data

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None



```python
import json
import boto3
import os
import re
from jsonschema import validate
from jsonschema.exceptions import ValidationError

# --- Configuration ---
PREPROCESSED_DIR = 'preprocessed_pdfs'
EXTRACTED_DIR = 'extracted_data'

# --- AWS Bedrock Client ---
bedrock = boto3.client('bedrock-runtime')

# --- Prompt Templates ---
def get_whitepaper_prompt_template():
    return """
Extract the following information from the provided model whitepaper text and return ONLY a JSON object. Do NOT include any introductory phrases, explanations, or any other text outside of the JSON object.

Here is the preprocessed text:
{preprocessed_text}

Extract:
- model_name: The name of the model.
- version: The version of the model.
- inputs: A list of model inputs (name, description, data type).
- outputs: A list of model outputs (name, description, data type).
- calculations: A list of calculations (step number, formula, description).
- assumptions: A list of assumptions.
- limitations: A list of limitations.

JSON format example (IMPORTANT: Return ONLY a JSON object like this, with NO other text):
{few_shot_example}
"""

def get_testplan_prompt_template():
      return """Extract the following from the model test plan and return ONLY a JSON object. Do NOT include any explanations or extra text.

Here is the preprocessed data:
{preprocessed_text}

Extract:
- test_plan_id: The ID of the test plan.
- objectives: A list of test objectives (objective_id, description).
- test_cases: A list of test cases (test_case_id, objective_id, description, input_data, expected_output).

Expected JSON format (return ONLY the JSON, nothing else):
{few_shot_example}
"""

def get_testresults_prompt_template():
    return """Extract the following and return ONLY a JSON object.  Do NOT include any other text.

Here are the preprocessed test results:
{preprocessed_text}

Extract:
- test_results_id: The ID of the test results set.
- results: A list of test results (test_case_id, status, actual_output, comments).

Return ONLY a JSON object in this format (NO other text):
{few_shot_example}
"""

def get_reconciliation_report_prompt_template():
    return """Extract the following and return ONLY a JSON object. Do NOT add any extra text.

Here is the preprocessed reconciliation report data:
{preprocessed_text}

Extract:
- reconciliation_report_id: The ID of the report.
- reconciled_elements: A list of reconciled data elements (data_element, source_system, target_system, status, discrepancy_amount, explanation).

Return a JSON object in EXACTLY this format (NO additional text):
{few_shot_example}
"""

# --- Few-Shot Examples ---
def get_whitepaper_few_shot_example():
    return """
{
  "model_name": "Advanced Loan Risk Assessment Model",
  "version": "2.1",
  "inputs": [
    {
      "name": "Borrower Annual Income",
      "description": "The annual income of the primary borrower, in USD.",
      "data_type": "float"
    },
    {
      "name": "Loan-to-Value Ratio",
      "description": "The ratio of the loan amount to the appraised value of the property.",
      "data_type": "float"
    },
    {
      "name": "Credit History Length",
      "description": "The length of the borrower's credit history, in years.",
      "data_type": "integer"
    },
    {
        "name": "Number of Open Credit Lines",
        "description": "Total number of currently open credit lines.",
        "data_type": "integer"
    }
  ],
  "outputs": [
    {
      "name": "Predicted Default Probability",
      "description": "The probability that the borrower will default on the loan, expressed as a decimal between 0 and 1.",
      "data_type": "float"
    },
    {
      "name": "Credit Risk Score",
      "description": "A numerical score representing the overall credit risk of the loan.",
      "data_type": "integer"
    },
     {
      "name": "Loan Recommendation",
      "description": "Recommendation for loan approval (Approve, Deny, Review).",
      "data_type": "string"
    }
  ],
  "calculations": [
    {
      "step": "1",
      "formula": "weighted_score = (income_weight * log(Borrower Annual Income + 1)) + (ltv_weight * Loan-to-Value Ratio) + (history_weight * Credit History Length)",
      "description": "Calculate a weighted score based on key input variables. A logarithmic transformation is applied to income."
    },
    {
      "step": "2",
      "formula": "default_probability = 1 / (1 + exp(-1 * (intercept + (weighted_score * score_coefficient))))",
      "description": "Calculate the probability of default using a logistic regression formula."
    }
  ],
  "assumptions": [
    {
      "description": "The relationship between the input variables and the probability of default is assumed to be stable over time."
    },
    {
      "description": "Borrowers with similar characteristics will exhibit similar default behavior."
    },
    {
      "description": "The model assumes that all input data is accurate and reliable."
    }
  ],
  "limitations": [
    {
      "description": "The model does not account for macroeconomic factors, such as changes in interest rates or unemployment."
    },
    {
      "description": "The model's accuracy may be limited for borrowers with very short or very long credit histories."
    },
    {
      "description": "The model is based on historical data and may not accurately predict future performance."
    }
  ]
}
    """

def get_testplan_few_shot_example():
    return """
{
    "test_plan_id": "TP_LoanRisk_v2.1",
    "objectives":[
        {"objective_id": "OBJ_1", "description": "Verify the correct calculation of the weighted score."},
        {"objective_id": "OBJ_2", "description": "Validate the default probability calculation across a range of input values."},
        {"objective_id": "OBJ_3", "description": "Ensure the model handles edge cases and invalid input data appropriately."}
    ],
    "test_cases":[
        {"test_case_id":"TC_1", "objective_id": "OBJ_1", "description": "Test with high income and low LTV.", "input_data": "Borrower Annual Income=150000, Loan-to-Value Ratio=0.6, Credit History Length=10,Number of Open Credit Lines=3", "expected_output": "weighted_score > 75"},
        {"test_case_id":"TC_2", "objective_id": "OBJ_1", "description": "Test with moderate income and moderate LTV.", "input_data": "Borrower Annual Income=75000, Loan-to-Value Ratio=0.8, Credit History Length=5, Number of Open Credit Lines=5", "expected_output": "50 <= weighted_score <= 75"},
        {"test_case_id":"TC_3", "objective_id": "OBJ_2", "description": "Test for low default probability.", "input_data": "Borrower Annual Income=200000, Loan-to-Value Ratio=0.5, Credit History Length=15, Number of Open Credit Lines=2", "expected_output": "Predicted Default Probability < 0.01"},
        {"test_case_id":"TC_4", "objective_id": "OBJ_3", "description": "Test with missing income data.", "input_data": "Borrower Annual Income=, Loan-to-Value Ratio=0.7, Credit History Length=8, Number of Open Credit Lines=2", "expected_output": "Error or appropriate handling of missing data"}
    ]
}
"""

def get_testresults_few_shot_example():
    return """
{
    "test_results_id": "TR_LoanRisk_v2.1",
    "results":[
        {"test_case_id":"TC_1", "status": "Pass", "actual_output": "weighted_score = 82.5", "comments": ""},
        {"test_case_id":"TC_2", "status": "Pass", "actual_output": "weighted_score = 63.2", "comments": ""},
        {"test_case_id":"TC_3", "status": "Pass", "actual_output": "Predicted Default Probability = 0.008", "comments": ""},
        {"test_case_id":"TC_4", "status": "Pass", "actual_output": "Error: Missing income data.", "comments": "Error handling as expected"}
    ]
}
"""
def get_reconciliation_report_few_shot_example():
    return """
{
  "reconciliation_report_id": "RR_LoanRisk_v2.1",
  "reconciled_elements": [
    {"data_element": "Borrower Annual Income", "source_system": "Loan Application", "target_system": "Model Input", "status": "Reconciled", "discrepancy_amount": null, "explanation": ""},
    {"data_element": "Loan-to-Value Ratio", "source_system": "Appraisal Report", "target_system": "Model Input", "status": "Reconciled", "discrepancy_amount": null, "explanation": ""},
    {"data_element": "Predicted Default Probability", "source_system": "Model Output", "target_system": "Risk Reporting System", "status": "Reconciled", "discrepancy_amount": null, "explanation": ""},
    {"data_element": "Credit Risk Score", "source_system": "Model Output", "target_system": "Risk Reporting System", "status": "Discrepancy", "discrepancy_amount": 5, "explanation": "Difference within acceptable tolerance (threshold = 10)."}
  ]
}
"""

# --- JSON Schema Validation (Optional but Recommended) ---

def get_whitepaper_output_schema():
    return {
      "type": "object",
      "properties": {
          "model_name": {"type": "string"},
          "version": {"type": "string"},
          "inputs": {
              "type": "array",
              "items": {
                  "type": "object",
                  "properties": {
                      "name": {"type": "string"},
                      "description": {"type": "string"},
                      "data_type": {"type": "string"}
                  },
                  "required": ["name", "description", "data_type"]
              }
          },
          "outputs": {
              "type": "array",
              "items": {
                  "type": "object",
                  "properties": {
                      "name": {"type": "string"},
                      "description": {"type": "string"},
                      "data_type": {"type": "string"}
                  },
                  "required": ["name", "description", "data_type"]
              }
          },
          "calculations": {
              "type": "array",
              "items": {
                "type": "object",
                "properties":{
                  "step":{"type":"string"},
                  "formula" : {"type":"string"},
                  "description":{"type":"string"}
                },
                "required": ["step", "formula", "description"]
              }
          },
           "assumptions": {
              "type": "array",
              "items": {
                "type":"object",
                "properties":{
                  "description":{"type":"string"}
                },
                "required":["description"]
              }
          },
          "limitations": {
              "type": "array",
              "items": {
                "type":"object",
                "properties":{
                  "description":{"type":"string"}
                },
                "required":["description"]
              }
          }
      },
      "required": ["model_name", "version", "inputs", "outputs"] # Example required fields
    }

def get_testplan_output_schema():
    return {
      "type": "object",
      "properties":{
        "test_plan_id":{"type":"string"},
        "objectives":{
            "type":"array",
            "items":{
                "type":"object",
                "properties":{
                    "objective_id":{"type":"string"},
                    "description":{"type":"string"}
                },
                "required":["objective_id","description"]
            }
        },
        "test_cases":{
            "type":"array",
            "items":{
                "type":"object",
                "properties":{
                    "test_case_id":{"type":"string"},
                    "objective_id":{"type":"string"},
                    "description":{"type":"string"},
                    "input_data":{"type":"string"},
                    "expected_output":{"type":"string"}
                },
                "required":["test_case_id","objective_id","description","input_data","expected_output"]
            }
        }
      },
      "required": ["test_plan_id", "objectives", "test_cases"]
    }

def get_testresults_output_schema():
    return {
      "type":"object",
      "properties":{
        "test_results_id":{"type":"string"},
        "results":{
            "type":"array",
            "items":{
                "type":"object",
                "properties":{
                    "test_case_id":{"type":"string"},
                    "status":{"type":"string"},
                    "actual_output":{"type":"string"},
                    "comments":{"type":"string"}
                },
                "required":["test_case_id","status"]
            }
        }
      },
      "required": ["test_results_id", "results"]
    }

def get_reconciliation_report_output_schema():
    return {
      "type":"object",
      "properties":{
        "reconciliation_report_id":{"type":"string"},
        "reconciled_elements":{
            "type":"array",
            "items":{
                "type":"object",
                "properties":{
                    "data_element":{"type":"string"},
                    "source_system":{"type":"string"},
                    "target_system":{"type":"string"},
                    "status":{"type":"string"},
                    "discrepancy_amount":{"type":["number", "null"]},
                    "explanation":{"type":"string"}
                },
                "required":["data_element","source_system","target_system","status"]
            }
        }
      },
      "required":["reconciliation_report_id","reconciled_elements"]
    }
# --- Core Processing Function ---

def process_file_chunk(file_path, artifact_type):
    """Processes a single chunk of a preprocessed file."""
    try:
        with open(file_path, 'r') as f:
            preprocessed_data = json.load(f)

        if artifact_type == 'whitepaper':
            prompt_template = get_whitepaper_prompt_template()
            few_shot_example = get_whitepaper_few_shot_example()
            schema = get_whitepaper_output_schema()
        elif artifact_type == 'testplan':
            prompt_template = get_testplan_prompt_template()
            few_shot_example = get_testplan_few_shot_example()
            schema = get_testplan_output_schema()
        elif artifact_type == 'testresults':
            prompt_template = get_testresults_prompt_template()
            few_shot_example = get_testresults_few_shot_example()
            schema = get_testresults_output_schema()
        elif artifact_type == 'reconciliationreport':
            prompt_template = get_reconciliation_report_prompt_template()
            few_shot_example = get_reconciliation_report_few_shot_example()
            schema = get_reconciliation_report_output_schema()
        else:
            raise ValueError(f"Invalid artifact type: {artifact_type}")

        prompt = prompt_template.format(preprocessed_text=preprocessed_data['text'], few_shot_example=few_shot_example)

        # print("----- PROMPT -----")  # Debugging print (Optional)
        # print(prompt)
        # print("------------------")

        response = bedrock.invoke_model(
            body=json.dumps({
                "prompt": f"\n\nHuman:{prompt}\n\nAssistant:",
                "max_tokens_to_sample": 4096,
                "temperature": 0.1,
                "top_p": 0.9,
            }),
            modelId="anthropic.claude-v3-sonnet",
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response.get('body').read())

        # print("----- RESPONSE BODY -----")  # Debugging print (Optional)
        # print(response_body)
        # print("-----------------------")

        extracted_data_text = response_body.get('completion')

        if extracted_data_text is None:
            print("Error: Claude 3 returned an empty completion.")
            return None

        try:
            extracted_data = json.loads(extracted_data_text)
        except json.JSONDecodeError as e:
            print(f"Error: JSONDecodeError: {e}")
            print(f"Invalid JSON: {extracted_data_text}")
            return None

        # JSON Schema Validation (Optional but Recommended)
        try:
            validate(instance=extracted_data, schema=schema)
        except ValidationError as e:
            print(f"Error: JSON Schema Validation Failed: {e}")
            print(f"For data: {extracted_data}")  # Print the data that failed validation
            return None

        return extracted_data

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
def extract_data_from_preprocessed_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            #  Simpler artifact type determination (since Step 1 now handles naming)
            match = re.match(r'(.+)_chunk_(\d+)\.json', filename)
            if not match:
                print(f"Skipping file with unexpected name format: {filename}")
                continue

            artifact_base_name = match.group(1)
            artifact_type = artifact_base_name.lower()

            extracted_data = process_file_chunk(file_path, artifact_type)


            if extracted_data:
                output_file_path = os.path.join(output_dir, filename)  # Same name
                with open(output_file_path, 'w') as outfile:
                    json.dump(extracted_data, outfile, indent=4)
                print(f"Extracted data from {filename}, saved to {output_file_path}")

    # --- Post-Processing (Combine Chunk Results) ---
    # After processing all chunks, combine the results for each artifact
    for artifact_base_name in set([f.split('_chunk_')[0] for f in os.listdir(output_dir) if f.endswith('.json')]):
      combine_chunks(output_dir, artifact_base_name)

def combine_chunks(extracted_dir, artifact_base_name):
    """Combines the extracted data from multiple chunks into a single JSON file."""
    combined_data = {}
    chunk_files = sorted([f for f in os.listdir(extracted_dir) if f.startswith(artifact_base_name) and f.endswith('.json')],
                         key=lambda x: int(x.split('_chunk_')[1].split('.json')[0])) # Sort by chunk number

    for chunk_file in chunk_files:
        file_path = os.path.join(extracted_dir, chunk_file)
        with open(file_path, 'r') as f:
            chunk_data = json.load(f)

            # Merge the chunk data into the combined data
            for key, value in chunk_data.items():
                if key not in combined_data:
                    combined_data[key] = value
                elif isinstance(value, list):
                    # Simple deduplication.
                    for item in value:
                        if item not in combined_data[key]:
                            combined_data[key].append(item)
                # Add other merging logic as needed (e.g., for strings, numbers)
    # Save the combined data
    output_file_path = os.path.join(extracted_dir, f"{artifact_base_name}.json")
    with open(output_file_path, 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)
    print(f"Combined data for {artifact_base_name} saved to {output_file_path}")

# --- Main Execution ---
extract_data_from_preprocessed_files(PREPROCESSED_DIR, EXTRACTED_DIR)
