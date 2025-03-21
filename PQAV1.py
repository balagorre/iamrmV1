def load_sources(cleaned_file_path, extracted_file_path):
    """
    Loads both the cleaned summary and full extracted content.
    
    Args:
        cleaned_file_path (str): Path to the JSON file containing cleaned results.
        extracted_file_path (str): Path to the file containing full extracted content.

    Returns:
        dict, str: Cleaned results as a dictionary and full extracted text as a string.
    """
    try:
        with open(cleaned_file_path, "r", encoding="utf-8") as f:
            cleaned_results = json.load(f)
        
        with open(extracted_file_path, "r", encoding="utf-8") as f:
            extracted_text = f.read()
        
        return cleaned_results, extracted_text
    except Exception as e:
        print(f"Error loading sources: {str(e)}")
        return None, None

# Example usage
cleaned_results, extracted_text = load_sources(
    "./cleaned_whitepaper_analysis.json",
    "./extracted_content/extracted_text.txt"
)


def search_full_extracted_content(query, extracted_text):
    """
    Searches the full extracted content for relevant sections based on keywords in the query.
    
    Args:
        query (str): User's query.
        extracted_text (str): Full text of the whitepaper.

    Returns:
        str: Relevant sections from the full extracted content.
    """
    # Split text into paragraphs for searching
    paragraphs = extracted_text.split("\n\n")
    
    # Find paragraphs containing keywords from the query
    keywords = query.lower().split()  # Split query into keywords
    relevant_paragraphs = [
        para for para in paragraphs if any(keyword in para.lower() for keyword in keywords)
    ]
    
    return "\n\n".join(relevant_paragraphs)

def search_all_sources(query, cleaned_results, extracted_text):
    """
    Searches both cleaned results and full extracted content for relevant sections.
    
    Args:
        query (str): User's query.
        cleaned_results (dict): Cleaned whitepaper analysis results.
        extracted_text (str): Full text of the whitepaper.

    Returns:
        dict: Combined relevant sections from both sources.
    """
    # Search in cleaned results
    relevant_sections = search_cleaned_results(query, cleaned_results)
    
    # Search in full extracted content
    additional_context = search_full_extracted_content(query, extracted_text)
    
    if additional_context:
        relevant_sections["Additional Context"] = additional_context
    
    return relevant_sections

# Example usage
query = "What are the model inputs?"
relevant_sections = search_all_sources(query, cleaned_results, extracted_text)
print(relevant_sections)


def ask_llm_with_combined_context(query, context):
    """
    Uses Claude via AWS Bedrock to answer a question based on combined context from all sources.
    
    Args:
        query (str): User's question.
        context (dict): Relevant sections retrieved from all sources.

    Returns:
        str: Answer generated by Claude.
    """
    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    # Format context into readable text
    formatted_context = "\n\n".join([f"{section}:\n{content}" for section, content in context.items()])
    
    prompt = f"""
    You are an expert tasked with answering a question based on the following context:

    {formatted_context}

    Question:
      {query}

    Provide a detailed and accurate answer. If information is missing or insufficient, state 'Not enough information available in provided context.'
    """
    
    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        return response_body["content"].strip()
    
    except Exception as e:
        logging.error(f"Error invoking LLM: {str(e)}")
        return f"Failed to generate answer: {str(e)}"

# Example usage
answer = ask_llm_with_combined_context(query, relevant_sections)
print(answer)

import ipywidgets as widgets
from IPython.display import display, clear_output

def interactive_qa_with_combined_sources(cleaned_results, extracted_text):
    """
    Creates an interactive Q&A interface for querying whitepaper analysis results using combined sources.
    
    Args:
        cleaned_results (dict): Cleaned whitepaper analysis results.
        extracted_text (str): Full text of the whitepaper.

    Returns:
        None
    """
    # Create widgets
    question_input = widgets.Text(
        value='',
        placeholder='Ask a question about the whitepaper',
        description='Question:',
        layout=widgets.Layout(width='80%')
    )
    
    submit_button = widgets.Button(
        description='Ask',
        button_style='primary',
        tooltip='Submit your question',
        icon='question'
    )
    
    output_area = widgets.Output()

    # Define button click event
    def on_submit_button_clicked(b):
        with output_area:
            clear_output()
            query = question_input.value.strip()
            if not query:
                print("Please enter a valid question.")
                return
            
            # Step 1: Search all sources for relevant sections
            relevant_sections = search_all_sources(query, cleaned_results, extracted_text)
            
            # Step 2: Invoke LLM with combined context
            answer = ask_llm_with_combined_context(query, relevant_sections)
            
            # Display answer and sources
            print("Answer:")
            print(answer)
            
            print("\nRelevant Sections:")
            for section, content in relevant_sections.items():
                print(f"{section}:")
                print(content)

    submit_button.on_click(on_submit_button_clicked)

    # Display interface
    display(widgets.HBox([question_input, submit_button]))
    display(output_area)

# Example usage
interactive_qa_with_combined_sources(cleaned_results, extracted_text)


