def analyze_query(user_query):
    """
    Analyzes a user's query to determine its type (general Q&A, validation, or raw data extraction).
    
    Args:
        user_query (str): The user's question.
        
    Returns:
        dict: A structured dictionary with query type, key terms, rephrased query, and related queries.
    """
    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    prompt = f"""
    You are an expert query analyst specializing in technical documentation and model validation tasks. 
    Analyze the following user query:

    USER QUERY: "{user_query}"

    Tasks:
    1. Identify the **type of query**: Is it general Q&A, validation-related, or raw data extraction?
       Examples:
       - General Q&A: "What is the purpose of this model?"
       - Validation-related: "Validate the model's accuracy metrics."
       - Raw data extraction: "List all model input attributes."
       
    2. Extract **key search terms** that would be most effective for finding relevant information (3-5 terms).
    3. Generate a **rephrased version** of the query optimized for semantic search.
    4. Create a list of **related questions** that might help expand the search scope.

    Format your response as JSON with these keys:
    {{
      "query_type": "<general|validation|raw_data>",
      "key_terms": ["<term_1>", "<term_2>", ...],
      "rephrased_query": "<optimized_query>",
      "related_queries": ["<related_question_1>", "<related_question_2>", ...]
    }}
    
    Return ONLY valid JSON output."""
    
    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
                "temperature": 0.2
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        analyzed_query = json.loads(response_body["content"])
        
        logging.info(f"Query analysis completed successfully: {analyzed_query}")
        return analyzed_query
    
    except Exception as e:
        logging.error(f"Error analyzing query: {str(e)}")
        
        # Fallback in case of error
        return {
            "query_type": "general",
            "key_terms": user_query.lower().split(),
            "rephrased_query": user_query,
            "related_queries": [user_query]
        }


 def process_question(user_query, cleaned_results):
    """
    Process a user question by analyzing it and routing it to the appropriate workflow.
    
    Args:
        user_query (str): The user's question.
        cleaned_results (dict): Cleaned summary document with structured sections.
        
    Returns:
        dict: Answer and metadata appropriate to the query type.
    """
    
    # Analyze the query
    analyzed_query = analyze_query(user_query)
    
    # Route based on query type
    if analyzed_query["query_type"] == "raw_data":
        logging.info(f"Processing raw data extraction query: {analyzed_query['rephrased_query']}")
        
        extracted_data = extract_raw_data(analyzed_query, cleaned_results)
        
        if extracted_data.get("error"):
            return {"error": extracted_data["error"]}
        
        return {"raw_data": extracted_data}
    
    elif analyzed_query["query_type"] == "validation":
        logging.info(f"Processing validation-related query: {analyzed_query['rephrased_query']}")
        
        # Call validation workflow here...
    
    else:
        logging.info(f"Processing general Q&A query: {analyzed_query['rephrased_query']}")
        
        # Call general Q&A workflow here...
def extract_raw_data(query_dict, cleaned_results):
    """
    Extracts raw data attributes based on user query.
    
    Args:
        query_dict (dict): Analyzed query with type and key terms.
        cleaned_results (dict): Cleaned summary document with structured sections.
        
    Returns:
        dict: Extracted raw data attributes or relevant sections.
    """
    key_terms = set(query_dict.get("key_terms", []))
    
    # Define section keywords for raw data extraction
    section_keywords = {
        "inputs": {"input", "inputs", "parameters", "variables", "attributes"},
        "outputs": {"output", "outputs", "results", "predictions"},
        # Add more sections as needed
    }
    
    extracted_data = {}
    
    for section_name, keywords in section_keywords.items():
        if any(term in key_terms for term in keywords):
            if section_name in cleaned_results and cleaned_results[section_name]:
                extracted_data[section_name] = cleaned_results[section_name]
    
    if not extracted_data:
        return {"error": f"No relevant raw data found for query: {query_dict.get('rephrased_query', '')}"}
    
    return extracted_data

def generate_followup_questions(query_dict, answer):
    """
    Generate relevant follow-up questions based on the user's query and answer.
    
    Args:
        query_dict (dict): Analyzed query with type and key terms.
        answer (str): The generated answer to the user's question.
        
    Returns:
        list: A list of suggested follow-up questions.
    """
    try:
        bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        prompt = f"""
        You are an expert tasked with generating follow-up questions based on a user's query and answer:

        ORIGINAL QUESTION: "{query_dict.get('rephrased_query', '')}"

        GENERATED ANSWER:
        {answer}

        Tasks:
          - Generate 3-5 follow-up questions that explore related aspects or clarify ambiguities.
          - Ensure questions are specific and actionable.

          Format your response as a single string containing all follow-up questions separated by newlines."""
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.7
            })
        )
        
        response_body = response['body'].read().decode('utf-8').strip()
        
        if not response_body:
            logging.error("Empty response received from LLM.")
            raise ValueError("Empty response received from LLM.")
        
        logging.info("Follow-up questions generated successfully.")
        return response_body.split("\n")
    
    except Exception as e:
        logging.error(f"Error generating follow-up questions: {str(e)}")
        
        # Fallback: Return a default follow-up question
        return ["What additional information would help validate this model?"]

