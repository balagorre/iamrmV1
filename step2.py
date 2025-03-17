import json
import boto3
import os
import re
from typing import List, Dict, Optional
from pydantic import BaseModel, ValidationError, Field

# --- Configuration ---
PREPROCESSED_DIR = 'preprocessed_pdfs'
EXTRACTED_DIR = 'extracted_data'
MAX_RETRIES = 3

# --- AWS Bedrock Client ---
bedrock = boto3.client('bedrock-runtime')

# --- Pydantic Models (Whitepaper Only - Modified) ---

class Input(BaseModel):
    name: str
    description: str
    data_type: str

class Output(BaseModel):
    name: str
    description: str
    data_type: str

class Calculation(BaseModel):
    step: str
    formula: str
    description: str

class Assumption(BaseModel):  # Optional section
    description: str

class Limitation(BaseModel):  # Optional section
    description: str

class Whitepaper(BaseModel):
    model_name: str
    version: str
    inputs: List[Input]
    outputs: List[Output]
    calculations: List[Calculation]
    model_performance: str  # Added Model Performance as mandatory (string for now)
    assumptions: Optional[List[Assumption]] = None  # Optional
    limitations: Optional[List[Limitation]] = None  # Optional

# --- Prompt Template (Whitepaper Only - Modified) ---

def get_whitepaper_prompt_template():
    return """
Extract the following information from the provided model whitepaper text and return ONLY a JSON object conforming to the specified format. Do NOT include any introductory phrases, explanations, or any other text outside of the JSON object.

Here is the preprocessed text:
{preprocessed_text}

Extract the following MANDATORY sections:
- model_name: The name of the model.
- version: The version of the model.
- inputs: A list of model inputs (name, description, data type).
- outputs: A list of model outputs (name, description, data type).
- calculations: A list of calculations (step number, formula, description).
- model_performance: A description of the model's performance, including key metrics and results.

You may ALSO extract the following OPTIONAL sections, if present:
- assumptions: A list of assumptions.
- limitations: A list of limitations.

Return ONLY a JSON object with the following structure (Do NOT include any other text):
{{
  "model_name": "<model name>",
  "version": "<version>",
  "inputs": [
    {{"name": "<input name>", "description": "<description>", "data_type": "<data type>"}},
    ...
  ],
  "outputs": [
    {{"name": "<output name>", "description": "<description>", "data_type": "<data type>"}},
    ...
  ],
  "calculations": [
    {{"step": "<step number>", "formula": "<formula>", "description": "<description>"}},
    ...
  ],
  "model_performance": "<description of model performance>",
  "assumptions": [  // Optional
    {{"description": "<assumption description>"}},
    ...
  ],
  "limitations": [  // Optional
    {{"description": "<limitation description>"}},
    ...
  ]
}}
"""
# --- Few-Shot Example (Whitepaper Only - Modified) ---
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
  "model_performance": "The model has an AUC of 0.85, a precision of 0.78, and a recall of 0.82 on the test dataset.  It demonstrates good predictive power and generalizability.",
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
# --- Core Processing Function (with Pydantic and Retry) ---

def process_file_chunk(file_path):
    """Processes a single whitepaper chunk, parses with Pydantic, retries."""
    try:
        with open(file_path, 'r') as f:
            preprocessed_data = json.load(f)

        prompt_template = get_whitepaper_prompt_template()
        few_shot_example = get_whitepaper_few_shot_example()
        model_class = Whitepaper  # Use the Whitepaper Pydantic model

        prompt = prompt_template.format(preprocessed_text=preprocessed_data['text'], few_shot_example=few_shot_example)
        retries = 0
        while retries < MAX_RETRIES:
            try:
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
                extracted_data_text = response_body.get('completion')

                if extracted_data_text is None:
                    raise ValueError("Claude 3 returned an empty completion.")

                # Parse and validate using Pydantic
                extracted_data = model_class.model_validate_json(extracted_data_text)
                return extracted_data

            except (ValueError, json.JSONDecodeError, ValidationError) as e:
                retries += 1
                error_message = str(e)
                print(f"Attempt {retries} failed for {file_path}: {error_message}")
                # Feed the error back to the LLM
                prompt = f"{prompt}\n\nAssistant: {extracted_data_text}\n\nHuman: That response was invalid.  Error: {error_message}.  Please correct the JSON and try again. Return ONLY the corrected JSON, and nothing else."

        print(f"Max retries exceeded for {file_path}.")
        return None

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_data_from_preprocessed_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            match = re.match(r'whitepaper_chunk_(\d+)\.json', filename)  # Only whitepaper
            if not match:
                print(f"Skipping file with unexpected name format: {filename}")
                continue

            extracted_data = process_file_chunk(file_path)  # No artifact_type needed

            if extracted_data:
                # Save as JSON (using Pydantic's .model_dump())
                output_file_path = os.path.join(output_dir, filename)
                with open(output_file_path, 'w') as outfile:
                    json.dump(extracted_data.model_dump(), outfile, indent=4)
                print(f"Extracted data from {filename}, saved to {output_file_path}")

    # Combine Chunk Results (for whitepaper only)
    combine_chunks(output_dir, "whitepaper")

def combine_chunks(extracted_dir, artifact_base_name):
    """Combines extracted data from multiple chunks into a single Pydantic object."""
    combined_data = None
    chunk_files = sorted(
        [f for f in os.listdir(extracted_dir) if f.startswith(artifact_base_name) and f.endswith('.json')],
        key=lambda x: int(x.split('_chunk_')[1].split('.json')[0])  # Sort by chunk number
    )

    # Use the Whitepaper Pydantic model
    model_class = Whitepaper

    for chunk_file in chunk_files:
        file_path = os.path.join(extracted_dir, chunk_file)
        with open(file_path, 'r') as f:
            try:
                chunk_data = model_class(**json.load(f))  # Create Pydantic object

                if combined_data is None:
                    combined_data = chunk_data
                else:
                    # Merge the chunk data into the combined data
                    for field in combined_data.model_fields:
                        chunk_value = getattr(chunk_data, field)
                        combined_value = getattr(combined_data, field)

                        if isinstance(combined_value, list):
                            if isinstance(chunk_value, list):
                                for item in chunk_value:
                                    if item not in combined_value:
                                        combined_value.append(item)
                        elif chunk_value is not None:
                            setattr(combined_data, field, chunk_value)

            except (ValidationError, TypeError) as e:
                print(f"Error loading or merging chunk {chunk_file}: {e}")
                return None  # Stop if a chunk has issues

    # Save the combined data
    if combined_data:
        output_file_path = os.path.join(extracted_dir, f"{artifact_base_name}.json")
        with open(output_file_path, 'w') as outfile:
            json.dump(combined_data.model_dump(), outfile, indent=4)
        print(f"Combined data for {artifact_base_name} saved to {output_file_path}")

# --- Main Execution ---
extract_data_from_preprocessed_files(PREPROCESSED_DIR, EXTRACTED_DIR)




########################################################################


import json
import boto3
import os
import re
from typing import List, Dict, Optional
from pydantic import BaseModel, ValidationError, Field

# --- Configuration ---
PREPROCESSED_DIR = 'preprocessed_pdfs'
EXTRACTED_DIR = 'extracted_data'
MAX_RETRIES = 3

# --- AWS Bedrock Client ---
bedrock = boto3.client('bedrock-runtime')

# --- Pydantic Models (Whitepaper Only - Modified) ---

class Input(BaseModel):
    name: str
    description: str
    data_type: str

class Output(BaseModel):
    name: str
    description: str
    data_type: str

class Calculation(BaseModel):
    step: str
    formula: str
    description: str

class Assumption(BaseModel):  # Optional section
    description: str

class Limitation(BaseModel):  # Optional section
    description: str

class Whitepaper(BaseModel):
    model_name: str
    version: str
    inputs: List[Input]
    outputs: List[Output]
    calculations: List[Calculation]
    model_performance: str  # Added Model Performance as mandatory (string for now)
    assumptions: Optional[List[Assumption]] = None  # Optional
    limitations: Optional[List[Limitation]] = None  # Optional

# --- Prompt Template (Whitepaper Only - Modified) ---

def get_whitepaper_prompt_template():
    return """
Extract the following information from the provided model whitepaper text and return ONLY a JSON object conforming to the specified format. Do NOT include any introductory phrases, explanations, or any other text outside of the JSON object.

Here is the preprocessed text:
{preprocessed_text}

Extract the following MANDATORY sections:
- model_name: The name of the model.
- version: The version of the model.
- inputs: A list of model inputs (name, description, data type).
- outputs: A list of model outputs (name, description, data type).
- calculations: A list of calculations (step number, formula, description).
- model_performance: A description of the model's performance, including key metrics and results.

You may ALSO extract the following OPTIONAL sections, if present:
- assumptions: A list of assumptions.
- limitations: A list of limitations.

Return ONLY a JSON object with the following structure (Do NOT include any other text):
{{
  "model_name": "<model name>",
  "version": "<version>",
  "inputs": [
    {{"name": "<input name>", "description": "<description>", "data_type": "<data type>"}},
    ...
  ],
  "outputs": [
    {{"name": "<output name>", "description": "<description>", "data_type": "<data type>"}},
    ...
  ],
  "calculations": [
    {{"step": "<step number>", "formula": "<formula>", "description": "<description>"}},
    ...
  ],
  "model_performance": "<description of model performance>",
  "assumptions": [  // Optional
    {{"description": "<assumption description>"}},
    ...
  ],
  "limitations": [  // Optional
    {{"description": "<limitation description>"}},
    ...
  ]
}}
"""
# --- Few-Shot Example (Whitepaper Only - Modified) ---
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
  "model_performance": "The model has an AUC of 0.85, a precision of 0.78, and a recall of 0.82 on the test dataset.  It demonstrates good predictive power and generalizability.",
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
# --- Core Processing Function (with Pydantic and Retry) ---

def process_file_chunk(file_path):
    """Processes a single whitepaper chunk, parses with Pydantic, retries."""
    try:
        with open(file_path, 'r') as f:
            preprocessed_data = json.load(f)

        prompt_template = get_whitepaper_prompt_template()
        few_shot_example = get_whitepaper_few_shot_example()
        model_class = Whitepaper  # Use the Whitepaper Pydantic model

        prompt = prompt_template.format(preprocessed_text=preprocessed_data['text'], few_shot_example=few_shot_example)
        retries = 0
        while retries < MAX_RETRIES:
            try:
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
                extracted_data_text = response_body.get('completion')

                if extracted_data_text is None:
                    raise ValueError("Claude 3 returned an empty completion.")

                # Parse and validate using Pydantic
                extracted_data = model_class.model_validate_json(extracted_data_text)
                return extracted_data

            except (ValueError, json.JSONDecodeError, ValidationError) as e:
                retries += 1
                error_message = str(e)
                print(f"Attempt {retries} failed for {file_path}: {error_message}")
                # Feed the error back to the LLM
                prompt = f"{prompt}\n\nAssistant: {extracted_data_text}\n\nHuman: That response was invalid.  Error: {error_message}.  Please correct the JSON and try again. Return ONLY the corrected JSON, and nothing else."

        print(f"Max retries exceeded for {file_path}.")
        return None

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
def extract_data_from_preprocessed_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- MODIFICATION FOR TESTING (Single File) ---
    filename = "whitepaper_chunk_0.json"  #  Hardcode the sample filename
    file_path = os.path.join(input_dir, filename)

    if not os.path.exists(file_path):
        print(f"Error: Sample file not found: {file_path}")
        return

    extracted_data = process_file_chunk(file_path)

    if extracted_data:
        output_file_path = os.path.join(output_dir, filename)
        with open(output_file_path, 'w') as outfile:
            json.dump(extracted_data.model_dump(), outfile, indent=4)
        print(f"Extracted data from {filename}, saved to {output_file_path}")

    # --- COMMENT OUT CHUNK COMBINATION FOR TESTING ---
    # combine_chunks(output_dir, "whitepaper")
    # --- END MODIFICATION ---

# --- MODIFICATION FOR TESTING (No Chunk Combination) ---
# The combine_chunks function is NOT needed for single-file testing.
# You can either comment it out or leave it as is (it won't be called).
def combine_chunks(extracted_dir, artifact_base_name):
  pass # do nothing
# --- Main Execution ---
extract_data_from_preprocessed_files(PREPROCESSED_DIR, EXTRACTED_DIR)
