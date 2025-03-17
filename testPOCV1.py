Step 1 (Preprocessing - Modified):



import json
import os
import re
from PyPDF2 import PdfReader

def preprocess_pdf(file_path, chunk_size=10000):
    # (This function remains the same as the PyPDF2 version with chunking)
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            all_chunks = []
            current_chunk = ""

            for page in reader.pages:
                text = page.extract_text()
                text = re.sub(r'\s+', ' ', text).strip()  # Clean text

                current_chunk += text + " "

                while len(current_chunk) >= chunk_size:
                    # Find a good place to split (e.g., sentence boundary)
                    split_index = current_chunk.rfind('. ', 0, chunk_size)
                    if split_index == -1:  # No period found
                        split_index = chunk_size

                    chunk_text = current_chunk[:split_index].strip()
                    all_chunks.append({"text": chunk_text})
                    current_chunk = current_chunk[split_index:].strip()

            # Add any remaining text
            if current_chunk:
                all_chunks.append({"text": current_chunk.strip()})

            return all_chunks

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
def process_local_pdfs(input_dir, output_dir):
    """Processes PDFs, creates chunked JSON files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            file_path = os.path.join(input_dir, filename)
            chunks = preprocess_pdf(file_path)

            if chunks:
                # --- MODIFICATION HERE ---
                #  Create a standardized base name for output files
                base_name = "whitepaper"  #  Or get it from a mapping (see below)

                for i, chunk in enumerate(chunks):
                    output_file_path = os.path.join(output_dir, f"{base_name}_chunk_{i}.json")
                    with open(output_file_path, 'w') as outfile:
                        json.dump(chunk, outfile, indent=4)
                    print(f"Processed chunk {i} of {filename}, saved to {output_file_path}")

# --- Example Usage ---
input_directory = 'input_pdfs'
output_directory = 'preprocessed_pdfs'

# --- (Optional: Filename to Artifact Type Mapping) ---
filename_to_artifact_type = {
    "System-Level White Paper.pdf": "whitepaper",
    "System Test Plan.pdf": "testplan", # Example
    "Test Results Summary.pdf" : "testresults", # Example
    "Reconciliation Report.pdf": "reconciliationreport" # Example
    # Add other mappings as needed
}

if not os.path.exists(input_directory):
    os.makedirs(input_directory)
# ... (Dummy PDF creation code - or use your real PDFs)

process_local_pdfs(input_directory, output_directory)



Modified Ste2:


import json
import boto3
import os
import re

# --- Configuration ---
PREPROCESSED_DIR = 'preprocessed_pdfs'
EXTRACTED_DIR = 'extracted_data'

# --- AWS Bedrock Client ---
bedrock = boto3.client('bedrock-runtime')

# --- Prompt Templates (Remain the same) ---
# ... (All your prompt template functions: get_whitepaper_prompt_template, etc.)

# --- Few-Shot Examples (Remain the same) ---
# ... (All your few-shot example functions)

def process_file_chunk(file_path, artifact_type):
     # (This function remains the same)
    try:
        with open(file_path, 'r') as f:
            preprocessed_data = json.load(f)

        # Simplified prompts, focusing on presence/absence in chunk
        if artifact_type == 'whitepaper':
            prompt_template = get_whitepaper_prompt_template()
            few_shot_example = get_whitepaper_few_shot_example()
        # ... (add elif blocks for other artifact types, with their own prompts)
        elif artifact_type == 'testplan':
            prompt_template = get_testplan_prompt_template()
            few_shot_example = get_testplan_few_shot_example()
        elif artifact_type == 'testresults':
            prompt_template = get_testresults_prompt_template()
            few_shot_example = get_testresults_few_shot_example()
        elif artifact_type == 'reconciliationreport':
            prompt_template = get_reconciliation_report_prompt_template()
            few_shot_example = get_reconciliation_report_few_shot_example()
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
        extracted_data_text = response_body.get('completion')
        extracted_data = json.loads(extracted_data_text)
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
            # --- MODIFICATION HERE ---
            #  Simpler artifact type determination (since Step 1 now handles naming)
            match = re.match(r'(.+)_chunk_(\d+)\.json', filename)
            if not match:
                print(f"Skipping file with unexpected name format: {filename}")
                continue

            artifact_base_name = match.group(1)
            # chunk_number = int(match.group(2)) # No longer needed here
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
    # (This function remains the same)
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
                     #Simple deduplication.
                    combined_data[key].extend(item for item in value if item not in combined_data[key])
                # Add other merging logic as needed (e.g., for strings, numbers)
    # Save the combined data
    output_file_path = os.path.join(extracted_dir, f"{artifact_base_name}.json")
    with open(output_file_path, 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)
    print(f"Combined data for {artifact_base_name} saved to {output_file_path}")

# --- Main Execution ---
extract_data_from_preprocessed_files(PREPROCESSED_DIR, EXTRACTED_DIR)


#Step 2 (Information Extraction - Modified):

import json
import boto3
import os
import re

# --- Configuration ---
PREPROCESSED_DIR = 'preprocessed_pdfs'
EXTRACTED_DIR = 'extracted_data'

# --- AWS Bedrock Client ---
bedrock = boto3.client('bedrock-runtime')

# --- Prompt Templates (Simplified - see explanation below) ---

def get_whitepaper_prompt_template():
    return """
You are a helpful assistant that extracts information from model documentation.

Here is a chunk of text from a model whitepaper:
{preprocessed_text}

Your task is to extract the following information *if present in this chunk*:
- model_name: The name of the model.
- version: The version of the model.
- inputs: Any model inputs mentioned. Each input should have a name, description, and data type.
- outputs: Any model outputs mentioned. Each output should have a name, description, and data type.
- calculations: Any calculations mentioned. Each calculation should include a step number, the formula, and a description.
- assumptions: Any assumptions mentioned.
- limitations: Any limitations mentioned.

Return your findings as a JSON object.  If a particular piece of information
is not found in this chunk, simply omit it from the JSON. Do *not* include empty lists.

Here's an example of the expected JSON output format (but remember, only include
information *actually found* in this chunk):

{few_shot_example}

Now, extract the information from the provided text chunk and return the JSON object.
"""

# --- Few-Shot Example (Simplified) ---
# The few-shot example should also reflect the chunk-based approach
def get_whitepaper_few_shot_example():
    return """
{
  "inputs": [
    {"name": "Applicant Age", "description": "Age of the loan applicant", "data_type": "integer"}
  ],
  "calculations": [
    {"step": "1", "formula": "score = (age_weight * Applicant Age) + (credit_score_weight * Credit Score)", "description": "Calculate a weighted score"}
  ]
}
"""

def process_file_chunk(file_path, artifact_type):
    """Processes a single chunk of a preprocessed file."""
    try:
        with open(file_path, 'r') as f:
            preprocessed_data = json.load(f)

        # Simplified prompts, focusing on presence/absence in chunk
        if artifact_type == 'whitepaper':
            prompt_template = get_whitepaper_prompt_template()
            few_shot_example = get_whitepaper_few_shot_example()
        # ... (add elif blocks for other artifact types, with their own prompts)
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
        extracted_data_text = response_body.get('completion')
        extracted_data = json.loads(extracted_data_text)
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
            # Extract artifact type and chunk number from filename
            match = re.match(r'(.+)_chunk_(\d+)\.json', filename) # Regex to match filename
            if not match:
                print(f"Skipping file with unexpected name format: {filename}")
                continue

            artifact_base_name = match.group(1)
            chunk_number = int(match.group(2))
            artifact_type = artifact_base_name.lower() #Infer type

            extracted_data = process_file_chunk(file_path, artifact_type)

            if extracted_data:
                output_file_path = os.path.join(output_dir, filename)
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
                     #Simple deduplication.
                    combined_data[key].extend(item for item in value if item not in combined_data[key])
                # Add other merging logic as needed (e.g., for strings, numbers)
    # Save the combined data
    output_file_path = os.path.join(extracted_dir, f"{artifact_base_name}.json")
    with open(output_file_path, 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)
    print(f"Combined data for {artifact_base_name} saved to {output_file_path}")

# --- Main Execution ---
extract_data_from_preprocessed_files(PREPROCESSED_DIR, EXTRACTED_DIR)


#
Okay, understood. Since you don't have access to Neptune and this is a POC, we'll use an in-memory graph library, networkx, to represent the relationships. This will allow you to demonstrate the traceability verification logic without requiring a database. networkx is a standard Python library for working with graphs.




import json
import os
import networkx as nx
import matplotlib.pyplot as plt  # For optional visualization


# --- Configuration ---
EXTRACTED_DIR = 'extracted_data'
REPORTS_DIR = 'reports'

def populate_graph(graph, whitepaper_data, testplan_data, testresults_data):
    """Populates the NetworkX graph with data."""

    # Add Model Node
    graph.add_node("Model", type='Model', name=whitepaper_data['model_name'], version=whitepaper_data['version'])

    # Add Requirement Nodes
    for req in whitepaper_data.get('inputs', []) + whitepaper_data.get('outputs', []):
        graph.add_node(req['name'], type='Requirement', description=req['description'], data_type=req.get('data_type', ""))
        graph.add_edge("Model", req['name'], type='DEFINES')

    # Add Assumption and Limitation Nodes
    for assumption in whitepaper_data.get('assumptions', []):
        graph.add_node(assumption['description'], type='Assumption')
        graph.add_edge("Model", assumption['description'], type='RELATES_TO')
    for limitation in whitepaper_data.get('limitations', []):
        graph.add_node(limitation['description'], type='Limitation')
        graph.add_edge("Model", limitation['description'], type='RELATES_TO')

    # Add Objective Nodes
    for obj in testplan_data.get('objectives', []):
        graph.add_node(obj['objective_id'], type='Objective', description=obj['description'])
        # No direct link from Model to Objective

        # Add Test Case Nodes and link to Objectives
        for tc in testplan_data.get('test_cases', []):
            graph.add_node(tc['test_case_id'], type='TestCase', description=tc['description'], objective_id=tc['objective_id'])
            graph.add_edge(obj['objective_id'], tc['test_case_id'], type='TESTS')

            # Add Test Result Nodes and link to Test Cases
            for tr in testresults_data.get('results', []):
                if tr['test_case_id'] == tc['test_case_id']:
                    graph.add_node(f"{tr['test_case_id']}_result", type='TestResult', status=tr['status'], actual_output=str(tr.get('actual_output', '')), test_case_id = tr['test_case_id'])
                    graph.add_edge(tc['test_case_id'], f"{tr['test_case_id']}_result", type='RESULTS_FOR')

    # Link Test Cases to Requirements (Simplified - NEEDS IMPROVEMENT)
    for tc in graph.nodes():
        if graph.nodes[tc]['type'] == 'TestCase':
            tc_description = graph.nodes[tc]['description']
            for req in graph.nodes():
                if graph.nodes[req]['type'] == 'Requirement':
                    req_name = graph.nodes[req]['name']
                    if req_name.lower() in tc_description.lower():
                        graph.add_edge(tc, req, type='COVERS')


def check_traceability(graph):
    """Performs traceability checks using NetworkX."""
    gaps = {}
    gaps['uncovered_requirements'] = []
    gaps['unlinked_test_cases'] = []
    gaps['failed_test_results'] = []

    # Requirements not covered by any test case
    for node in graph.nodes():
        if graph.nodes[node]['type'] == 'Requirement':
            covered = False
            for neighbor in graph.neighbors(node):
                if graph.nodes[neighbor]['type'] == 'TestCase' and any(graph[neighbor][node]['type'] == 'COVERS' for neighbor, node in graph.edges(neighbor, data=True)):
                  covered = True
                  break
            if not covered:
                gaps['uncovered_requirements'].append(graph.nodes[node])

    # Test cases not linked to any objective
    for node in graph.nodes():
        if graph.nodes[node]['type'] == 'TestCase':
            linked = False
            for neighbor in graph.neighbors(node):
                if graph.nodes[neighbor]['type'] == 'Objective'and any(graph[node][neighbor]['type'] == 'TESTS' for neighbor, node in graph.edges(node, data=True)):
                    linked = True
                    break
            if not linked:
                gaps['unlinked_test_cases'].append(graph.nodes[node])

    # Find failed test results.
    for node in graph.nodes():
      if graph.nodes[node]['type'] == 'TestResult':
        if graph.nodes[node]['status'].lower() == 'fail':
          gaps['failed_test_results'].append(graph.nodes[node])

    return gaps

def visualize_graph(graph, output_file="graph.png"):
    """Visualizes the NetworkX graph (optional)."""
    try:
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(graph, seed=42)  # You can use other layouts too

        # Draw different node types with different colors
        node_colors = {
            'Model': 'red',
            'Requirement': 'blue',
            'Assumption': 'green',
            'Limitation': 'cyan',
            'Objective': 'orange',
            'TestCase': 'purple',
            'TestResult': 'pink',
        }

        for node_type, color in node_colors.items():
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[node for node, data in graph.nodes(data=True) if data.get('type') == node_type],
                node_color=color,
                label=node_type,
            )

        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos)
        plt.legend()
        plt.savefig(output_file)
        print(f"Graph visualization saved to {output_file}")

    except Exception as e:
        print(f"Error during visualization: {e}.  Make sure you have matplotlib installed.")


def main():
    """Main function to perform traceability verification."""

    # 1. Create an empty NetworkX graph
    graph = nx.DiGraph()  # Use a directed graph

    # 2. Load Extracted Data
    whitepaper_file = os.path.join(EXTRACTED_DIR, 'whitepaper.json')
    testplan_file = os.path.join(EXTRACTED_DIR, 'testplan.json')
    testresults_file = os.path.join(EXTRACTED_DIR, 'testresults.json')

    with open(whitepaper_file, 'r') as f:
        whitepaper_data = json.load(f)
    with open(testplan_file, 'r') as f:
        testplan_data = json.load(f)
    with open(testresults_file, 'r') as f:
        testresults_data = json.load(f)

    # 3. Populate the Graph
    populate_graph(graph, whitepaper_data, testplan_data, testresults_data)

    # 4. Perform Traceability Checks
    traceability_gaps = check_traceability(graph)
    print("Traceability Gaps:", traceability_gaps)

    # 5. Generate Report
    report = {
        "traceability_gaps": traceability_gaps
    }
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
    report_file = os.path.join(REPORTS_DIR, 'traceability_report.json')
    with open(report_file, 'w') as outfile:
        json.dump(report, outfile, indent=4)
    print(f"Traceability report saved to {report_file}")

    # 6. Visualize the graph (optional)
    visualize_graph(graph)


if __name__ == '__main__':
    main()
    #To install Matplotlib, run !pip install matplotlib in your notebook


Okay, let's move on to Step 4: Reconciliation Validation (SageMaker Notebook). This step will focus on:

Loading the extracted JSON data from Step 2, specifically the data from the whitepaper and the reconciliation report.
Using Claude 3 to compare the "outputs" section of the whitepaper with the "reconciled_elements" of the reconciliation report.
Identifying any discrepancies:
Outputs listed in the whitepaper that are not present in the reconciliation report.
Elements in the reconciliation report that are not listed as outputs in the whitepaper.
Generating a report summarizing the discrepancies.
Here's the code for Step 4:

import json
import boto3
import os

# --- Configuration ---
EXTRACTED_DIR = 'extracted_data'
REPORTS_DIR = 'reports'

# --- AWS Bedrock Client ---
bedrock = boto3.client('bedrock-runtime')

# --- Prompt Template ---
def get_reconciliation_validation_prompt():
    return """
You are a helpful assistant that validates reconciliation reports against model documentation.

Here is the extracted JSON data from the model whitepaper:
{whitepaper_json}

And here is the extracted JSON data from the reconciliation report:
{reconciliation_report_json}

Your task is to compare the 'outputs' section of the whitepaper with the 'reconciled_elements' section of the reconciliation report.  Identify any data elements that are listed as outputs in the whitepaper but are NOT present in the reconciliation report. Also identify any elements in the reconciliation report that are not listed as outputs in the whitepaper.

Return your findings as a JSON object with the following structure:

{
    "missing_from_reconciliation": [
        {"data_element": "...", "description": "..."}
        ...
    ],
    "extra_in_reconciliation": [
       {"data_element": "...", "description": "..."}
        ...
    ]
}

Here's a few-shot example to demonstrate:
{few_shot_example}

Now, perform the comparison and return the JSON object.
"""

# --- Few-Shot Example ---
def get_reconciliation_validation_few_shot_example():
    return """
{
  "missing_from_reconciliation": [
    {"data_element": "Loan Status", "description": "Approved or Rejected"}
  ],
  "extra_in_reconciliation": [
    {"data_element": "Customer ID", "description": "Internal customer identifier"}
  ]
}
"""

def validate_reconciliation(whitepaper_data, reconciliation_report_data):
    """
    Performs reconciliation validation using Claude 3.

    Args:
        whitepaper_data (dict): Extracted data from the whitepaper.
        reconciliation_report_data (dict): Extracted data from the reconciliation report.

    Returns:
        dict: A dictionary containing the discrepancies, or None on error.
    """
    try:
        prompt = get_reconciliation_validation_prompt().format(
            whitepaper_json=json.dumps(whitepaper_data),
            reconciliation_report_json=json.dumps(reconciliation_report_data),
            few_shot_example=get_reconciliation_validation_few_shot_example()
        )

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
        discrepancies_text = response_body.get('completion')
        discrepancies = json.loads(discrepancies_text)

        return discrepancies

    except Exception as e:
        print(f"Error during reconciliation validation: {e}")
        return None

def main():
    """Main function to perform reconciliation validation."""

    # 1. Load Extracted Data
    whitepaper_file = os.path.join(EXTRACTED_DIR, 'whitepaper.json')
    reconciliation_report_file = os.path.join(EXTRACTED_DIR, 'reconciliationreport.json')

    try:
        with open(whitepaper_file, 'r') as f:
            whitepaper_data = json.load(f)
        with open(reconciliation_report_file, 'r') as f:
            reconciliation_report_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find required input files: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: JSON parsing error in input files: {e}")
        return

    # 2. Perform Reconciliation Validation
    discrepancies = validate_reconciliation(whitepaper_data, reconciliation_report_data)

    if discrepancies:
        # 3. Generate Report
        print("Reconciliation Discrepancies:", discrepancies)

        if not os.path.exists(REPORTS_DIR):
            os.makedirs(REPORTS_DIR)
        report_file = os.path.join(REPORTS_DIR, 'reconciliation_report.json')
        with open(report_file, 'w') as outfile:
            json.dump(discrepancies, outfile, indent=4)
        print(f"Reconciliation report saved to {report_file}")
    else:
        print("No discrepancies found or an error occurred.")
if __name__ == '__main__':
    main()
