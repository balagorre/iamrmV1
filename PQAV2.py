# Complete Function Invocation Flow for Enhanced Q&A System

Below is the comprehensive flow of all function invocations in our enhanced system, showing the exact sequence in which functions are called and how they interact when processing queries.

---

## 1. Main Entry Point

```
run_qa_cli()
  │
  ├── initialize_globals() ───> Initialize global variables
  │   │
  │   └── load_sources(cleaned_file_path, extracted_file_path) ───> Load whitepaper data
  │
  ├── setup_bedrock_client() ───> Connect to AWS Bedrock
  │
  └── process_question(query) ───> Route query to appropriate workflow
      │
      ├── analyze_validation_query(query) ───> Determine query type
      │
      ├── IF validation query ───> answer_validation_query(query, cleaned_results, extracted_text)
      │
      └── IF general query ───> answer_question(query)
```

---

## 2. Validation Query Path (Complete)

```
answer_validation_query(query, cleaned_results, extracted_text)
  │
  ├── analyze_validation_query(query) ───> Extracts intent, key terms, rephrased query, and related queries
  │   │
  │   └── [Claude LLM Call] ───> Analyze query intent
  │
  ├── enhanced_technical_search(query_dict, cleaned_results, extracted_text) ───> Retrieves relevant sections
  │   │
  │   ├── search_sources(rephrased_query) ───> Search structured data
  │   │   │
  │   │   └── section_keywords matching ───> Match query terms to sections
  │   │
  │   ├── semantic_search(rephrased_query, paragraphs) ───> Perform vector search
  │   │   │
  │   │   ├── TfidfVectorizer() ───> Create vectors
  │   │   └── cosine_similarity() ───> Calculate relevance
  │   │
  │   └── semantic_search(related_queries, paragraphs) ───> Expand search with related queries
  │
  ├── refine_context_safely(query, search_results) ───> Refines context for LLM processing
  │   │
  │   ├── Prioritize technical sections ───> Order sections by relevance
  │   │
  │   ├── Truncate sections to fit token limits ───> Manage context size
  │   │
  │   └── [Claude LLM Call] ───> Optimize context for the query
  │
  ├── generate_validated_answer(query, refined_context) ───> Generates structured validation answer
  │   │
  │   ├── Detect validation aspects ───> Identify validation focus
  │   │
  │   ├── Select specialized prompt ───> Choose prompt by validation type
  │   │
  │   └── [Claude LLM Call] ───> Generate structured answer
  │
  ├── verify_technical_accuracy(query, detailed_answer, refined_context) ───> Verifies technical accuracy
  │   │
  │   ├── Extract numerical values and formulas ───> Identify technical elements
  │   │
  │   ├── Prepare verification prompt ───> Create prompt for accuracy check
  │   │
  │   ├── [Claude LLM Call] ───> Verify answer against context
  │   │
  │   └── Parse response body and extract verification result ───> Process LLM output
  │
  ├── IF verification has critical issues:
  │   └── apply_technical_corrections(detailed_answer, verification) ───> Corrects critical issues
  │       │
  │       └── Add correction warnings and issue summary ───> Annotate answer with corrections
  │
  ├── format_answer_with_citations(corrected_answer, search_results) ───> Adds source citations
  │   │
  │   ├── Extract section texts ───> Prepare citation sources
  │   │
  │   ├── Truncate context if too large ───> Manage token limits
  │   │
  │   └── [Claude LLM Call] ───> Add citations to answer
  │
  ├── calculate_technical_confidence(verification) ───> Calculates confidence score
  │   │
  │   └── Adjust score based on issue counts ───> Compute final confidence
  │
  ├── determine_validation_status(verification) ───> Determines overall validation status
  │   │
  │   └── Check critical issue count ───> Assign appropriate status
  │
  └── generate_validation_followups(query, corrected_answer) ───> Generates follow-up questions
      │
      └── [Claude LLM Call] ───> Generate technical follow-up questions
```

---

## 3. General Query Path (Complete)

```
answer_question(query)
  │
  ├── analyze_and_transform_query(query) ───> Extracts intent, key terms, rephrased query, and related queries
  │   │
  │   └── [Claude LLM Call] ───> Analyze and transform query
  │
  ├── enhanced_semantic_search(query_dict) ───> Retrieves relevant sections
  │   │
  │   ├── search_sources(rephrased_query) ───> Search structured data
  │   │   │
  │   │   └── section_keywords matching ───> Match query terms to sections
  │   │
  │   ├── search_full_extracted_content(rephrased_query) ───> Search unstructured text
  │   │   │
  │   │   └── semantic_search(query, paragraphs) ───> Vector search
  │   │       │
  │   │       ├── TfidfVectorizer() ───> Create vectors
  │   │       └── cosine_similarity() ───> Calculate relevance
  │   │
  │   └── Expand search with related queries ───> Add context from related queries
  │
  ├── refine_context(query, search_results) ───> Refines context for LLM processing
  │   │
  │   ├── Format context sections ───> Prepare context for LLM
  │   │
  │   ├── Check context size ───> Manage token limits
  │   │
  │   └── [Claude LLM Call] ───> Optimize context for the query
  │
  ├── generate_chain_of_thought_answer(query, refined_context) ───> Generates structured answer
  │   │
  │   └── [Claude LLM Call] ───> Generate answer with chain-of-thought reasoning
  │
  ├── evaluate_answer_quality(query, detailed_answer, refined_context) ───> Evaluates answer quality
  │   │
  │   └── [Claude LLM Call] ───> Evaluate accuracy, completeness, and quality
  │
  ├── IF evaluation.score  Improves answer
  │       │
  │       └── [Claude LLM Call] ───> Generate improved answer based on evaluation
  │
  └── generate_followup_questions(query, final_answer) ───> Generates follow-up questions
      │
      └── [Claude LLM Call] ───> Generate general follow-up questions
```

---

## 4. Shared Utility Functions

```
format_content(content)
  └── Format different content types (lists, dictionaries, strings) for inclusion in prompts

semantic_search(query, text_chunks, top_k=5)
  ├── TfidfVectorizer() ───> Create term frequency-inverse document frequency vectors
  ├── cosine_similarity() ───> Calculate similarity between vectors
  └── Return top_k most similar chunks with scores
```

---

## 5. Function Invocation Flow in Context Management

```
refine_context_safely(query, raw_context)
  │
  ├── Detect technical validation query ───> Check for technical terms
  │
  ├── Prioritize sections based on query type ───> Order sections by relevance
  │
  ├── Format sections in priority order ───> Prepare context
  │   │
  │   └── format_content(content) ───> Format different content types
  │
  ├── Truncate context to fit token limits ───> Manage size
  │
  └── [Claude LLM Call] ───> Optimize context for query
```

---

## 6. LLM Response Handling Flow

```
[Claude LLM Call]
  │
  ├── bedrock_client.invoke_model() ───> Send prompt to AWS Bedrock
  │
  ├── Extract response_body ───> Parse API response
  │
  ├── Try to parse JSON (if applicable) ───> Process structured output
  │
  ├── Validate response structure ───> Check for expected fields
  │
  └── Return processed response or handle errors ───> Provide results or fallback
```

---

This comprehensive flow diagram captures all functions in the enhanced system and their exact sequence of invocation. The system routes queries to either the specialized validation path or the general query path, with each path using specialized functions optimized for its particular use case.

The validation path focuses on technical accuracy, citation, and verification, while the general path focuses on completeness and clarity. Both paths leverage advanced LLM capabilities through prompt engineering and strategic context management.

---
Answer from Perplexity: pplx.ai/share
















#################

def verify_technical_accuracy(query, answer, context):
    """
    Verify the technical accuracy of model validation answers by comparing against source context.
    
    Args:
        query (str): The original user query.
        answer (str): The generated answer to verify.
        context (str): The source context used to generate the answer.
        
    Returns:
        dict: Comprehensive verification results with detailed accuracy assessment.
    """
    try:
        # STEP 1: Prepare verification prompt
        verification_prompt = f"""
        You are an expert technical reviewer verifying the accuracy of a model validation answer.
        
        ORIGINAL QUESTION: {query}
        
        ANSWER TO VERIFY:
        {answer}
        
        SOURCE CONTEXT:
        {context}
        
        Perform a thorough technical accuracy review focusing on these aspects:
        
        1. NUMERICAL ACCURACY: Verify all numerical values match the source context.
        2. FORMULA ACCURACY: Verify all formulas and equations match the source.
        3. METHODOLOGICAL ACCURACY: Verify descriptions of methods/processes are accurate.
        4. CITATION ACCURACY: Verify references to sections or sources are correct.
        5. COMPLETENESS: Verify that important context information was not omitted.
        
        For each issue found, classify it by severity:
        - CRITICAL: Factually incorrect numbers, formulas, or methodological claims that would mislead.
        - IMPORTANT: Significant omissions, unclear explanations, or imprecise descriptions.
        - MINOR: Style issues, minor imprecisions, or areas that could be clarified.
        
        For each issue, provide:
        - Specific text from the answer that contains the issue.
        - The correct information from the context.
        - Clear correction suggestion.
        
        Format your response as a JSON object with these keys:
        {{
          "verified_accurate": boolean,
          "issues_found": integer,
          "critical_issues": [{{"issue": "<description>", "correction": "<suggestion>", "location": "<location>"}}],
          "important_issues": [{{"issue": "<description>", "correction": "<suggestion>", "location": "<location>"}}],
          "minor_issues": [{{"issue": "<description>", "correction": "<suggestion>", "location": "<location>"}}],
          "verification_summary": "<brief_summary>",
          "suggested_corrections": ["<correction_1>", "<correction_2>", ...]
        }}
        
        Return only valid JSON output."""
        
        # STEP 2: Call Claude via AWS Bedrock
        bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": verification_prompt}],
                "max_tokens": 2500,
                "temperature": 0.2
            })
        )
        
        # STEP 3: Parse response
        response_body = response['body'].read().decode('utf-8').strip()
        
        if not response_body:
            logging.error("Empty response received from LLM.")
            raise ValueError("Empty response received from LLM.")
        
        try:
            verification_result = json.loads(response_body)
            
            # Validate structure
            if not isinstance(verification_result, dict):
                logging.error("Verification result is not a valid dictionary.")
                raise ValueError("Invalid verification result format.")
            
            # Ensure required keys exist
            required_keys = ["verified_accurate", "issues_found", "critical_issues", 
                             "important_issues", "minor_issues", 
                             "verification_summary", "suggested_corrections"]
            
            for key in required_keys:
                if key not in verification_result:
                    logging.warning(f"Missing key '{key}' in verification result.")
                    verification_result[key] = None
            
            logging.info(f"Technical accuracy verification completed successfully.")
            return verification_result
        
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON received from LLM: {response_body}")
            raise ValueError(f"Invalid JSON received from LLM.") from e
    
    except Exception as e:
        logging.error(f"Error during technical accuracy verification: {str(e)}")
        
        # Fallback result for error handling
        return {
            "verified_accurate": False,
            "issues_found": 1,
            "critical_issues": [{"issue": f"Verification failed due to technical error: {str(e)}", 
                                "correction": "Manual review required", 
                                "location": "entire document"}],
            "important_issues": [],
            "minor_issues": [],
            "verification_summary": "Technical error during verification prevented detailed analysis.",
            "suggested_corrections": []
        }

















def generate_validation_followups(query, answer):
    """
    Generate relevant follow-up questions based on the user's query and the generated answer.
    
    Args:
        query (str): The original user query.
        answer (str): The generated answer to the query.
        
    Returns:
        str: A single string containing follow-up questions.
    """
    try:
        # Prompt for generating follow-up questions
        prompt = f"""
        You are an expert model validator tasked with generating follow-up questions to help users explore 
        related aspects of a technical validation task. Analyze the following:

        ORIGINAL QUESTION: "{query}"

        GENERATED ANSWER:
        {answer}

        Tasks:
        1. Identify any ambiguities or areas where additional information might be helpful.
        2. Generate 3-5 follow-up questions that:
           - Explore related aspects not covered in the original answer.
           - Dig deeper into specifics mentioned in the answer (e.g., methodologies, assumptions, performance metrics).
           - Clarify potential ambiguities or explore edge cases.

        Instructions:
        - Be specific and concise in your questions.
        - Ensure the questions are relevant to the original query and answer.
        - Avoid redundant or overly generic questions.

        Format your response as a single string containing all follow-up questions, separated by newlines."""
        
        # Call Claude via AWS Bedrock
        bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.7
            })
        )
        
        # Parse response
        response_body = response['body'].read().decode('utf-8').strip()
        
        # Check if response is empty
        if not response_body:
            logging.error("Empty response received from LLM.")
            raise ValueError("Empty response received from LLM.")
        
        logging.info("Follow-up questions generated successfully.")
        return response_body
    
    except Exception as e:
        logging.error(f"Error generating follow-up questions: {str(e)}")
        
        # Fallback: Return a default follow-up question
        return "What additional information would help validate this model?"







def format_answer_with_citations(answer, raw_context):
    """
    Add precise source citations to answers for auditability, with context size management.
    
    Args:
        answer (str): The generated answer to enhance with citations.
        raw_context (dict): The raw search results containing relevant sections and sources.
        
    Returns:
        str: Enhanced answer with citations added in square brackets.
    """
    try:
        # STEP 1: Extract and prioritize sections
        prioritized_sections = [
            "Calculations", "Model Performance", "Testing Summary",
            "Inputs", "Outputs", "Reconciliation", "Solution Specification"
        ]
        
        # Filter and prioritize sections from raw_context
        prioritized_context = {section: raw_context[section] for section in prioritized_sections if section in raw_context}
        
        # STEP 2: Manage context size
        max_context_chars = 80000  # ~20K tokens, leaving room for instructions and answer
        current_length = 0
        truncated_context = {}
        
        for section_name, content in prioritized_context.items():
            if isinstance(content, list):
                # Handle lists (e.g., structured data or multiple paragraphs)
                section_text = "\n".join(content)
            else:
                # Handle single string content
                section_text = content
            
            # Check if adding this section would exceed the limit
            if current_length + len(section_text) > max_context_chars:
                # Truncate this section to fit within the limit
                available_space = max_context_chars - current_length - len(section_name) - 50  # Add buffer for formatting
                truncated_section = section_text[:available_space] + "\n[Truncated due to length]"
                truncated_context[section_name] = truncated_section
                current_length += len(truncated_section)
                break
            else:
                # Add full section if it fits
                truncated_context[section_name] = section_text
                current_length += len(section_text)
        
        logging.info(f"Final context size: {current_length} characters across {len(truncated_context)} sections.")
        
        # STEP 3: Prepare citation prompt with truncated context
        citation_prompt = f"""
        You are an expert tasked with enhancing a technical answer by adding precise source citations.

        ANSWER:
        {answer}
        
        SOURCE SECTIONS:
        {json.dumps(truncated_context, indent=2)}
        
        Instructions:
        1. Review the answer and identify technical claims, numbers, or methodologies.
        2. For each significant claim, add a precise citation in square brackets (e.g., [Calculations], [Model Performance]).
        3. Citations should reference the source section (e.g., "Calculations", "Model Performance").
        4. DO NOT change any technical details in the original answer.
        5. Ensure every numerical value and technical assertion has a citation.
        
        Return the enhanced answer with added citations in square brackets."""
        
        # STEP 4: Call Claude via AWS Bedrock
        bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": citation_prompt}],
                "max_tokens": 2500,
                "temperature": 0.2
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        cited_answer = response_body["content"].strip()
        
        # STEP 5: Validate output
        if "[" not in cited_answer or "]" not in cited_answer:
            logging.warning("Citations were not added to the answer. Returning original answer.")
            return answer
        
        logging.info("Citations successfully added to the answer.")
        return cited_answer
    
    except Exception as e:
        logging.error(f"Error adding citations: {str(e)}")
        
        # Fallback: Return original answer if citation process fails
        return f"""
        [NOTE: Failed to add citations due to technical error.]
        
        Original Answer:
        {answer}
        
        Error Details:
        {str(e)}
        """


















def generate_validation_followups(query, answer):
    """
    Generate relevant follow-up questions based on the user's query and the generated answer.
    
    Args:
        query (str): The original user query.
        answer (str): The generated answer to the query.
        
    Returns:
        list: A list of suggested follow-up questions.
    """
    try:
        # Prompt for generating follow-up questions
        prompt = f"""
        You are an expert model validator tasked with generating follow-up questions to help users explore 
        related aspects of a technical validation task. Analyze the following:

        ORIGINAL QUESTION: "{query}"

        GENERATED ANSWER:
        {answer}

        Tasks:
        1. Identify any ambiguities or areas where additional information might be helpful.
        2. Generate 3-5 follow-up questions that:
           - Explore related aspects not covered in the original answer.
           - Dig deeper into specifics mentioned in the answer (e.g., methodologies, assumptions, performance metrics).
           - Clarify potential ambiguities or explore edge cases.

        Instructions:
        - Be specific and concise in your questions.
        - Ensure the questions are relevant to the original query and answer.
        - Avoid redundant or overly generic questions.

        Format your response as a JSON array of strings, each containing one follow-up question."""
        
        # Call Claude via AWS Bedrock
        bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.7,
                "response_format": {"type": "json_object"}
            })
        )
        
        # Parse response
        response_body = json.loads(response['body'].read().decode('utf-8'))
        followup_questions = json.loads(response_body["content"])
        
        # Validate output
        if not isinstance(followup_questions, list):
            logging.warning("Follow-up questions were not returned as a list. Returning default question.")
            return ["What additional information would help validate this model?"]
        
        logging.info(f"Generated {len(followup_questions)} follow-up questions.")
        return followup_questions
    
    except Exception as e:
        logging.error(f"Error generating follow-up questions: {str(e)}")
        
        # Fallback: Return a default follow-up question
        return ["What additional information would help validate this model?"]



def format_answer_with_citations(answer, raw_context):
    """
    Add precise source citations to answers for auditability.
    
    Args:
        answer (str): The generated answer to enhance with citations.
        raw_context (dict): The raw search results containing relevant sections and sources.
        
    Returns:
        str: Enhanced answer with citations added in square brackets.
    """
    try:
        # STEP 1: Extract text from each section for citation purposes
        section_texts = {}
        
        for section_name, content in raw_context.items():
            if isinstance(content, list):
                if all(isinstance(item, dict) for item in content):
                    # Handle structured data (e.g., inputs/outputs)
                    section_texts[section_name] = [f"{item.get('name', '')}: {item.get('description', '')}" 
                                                   for item in content]
                else:
                    # Handle list of strings
                    section_texts[section_name] = content
            else:
                # Handle single string content
                section_texts[section_name] = [content]
        
        # STEP 2: Prepare citation prompt
        citation_prompt = f"""
        You are an expert tasked with enhancing a technical answer by adding precise source citations.

        ANSWER:
        {answer}
        
        SOURCE SECTIONS:
        {json.dumps(section_texts, indent=2)}
        
        Instructions:
        1. Review the answer and identify technical claims, numbers, or methodologies.
        2. For each significant claim, add a precise citation in square brackets (e.g., [Calculations], [Model Performance]).
        3. Citations should reference the source section (e.g., "Calculations", "Model Performance").
        4. DO NOT change any technical details in the original answer.
        5. Ensure every numerical value and technical assertion has a citation.
        
        Return the enhanced answer with added citations in square brackets."""
        
        # STEP 3: Call Claude via AWS Bedrock
        bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": citation_prompt}],
                "max_tokens": 2500,
                "temperature": 0.2
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        cited_answer = response_body["content"].strip()
        
        # STEP 4: Validate output
        if "[" not in cited_answer or "]" not in cited_answer:
            logging.warning("Citations were not added to the answer. Returning original answer.")
            return answer
        
        logging.info("Citations successfully added to the answer.")
        return cited_answer
    
    except Exception as e:
        logging.error(f"Error adding citations: {str(e)}")
        
        # Fallback: Return original answer if citation process fails
        return f"""
        [NOTE: Failed to add citations due to technical error.]
        
        Original Answer:
        {answer}
        
        Error Details:
        {str(e)}
        """
























# Global variables for cleaned results and extracted text
cleaned_results = None
extracted_text = None

def load_sources(cleaned_file_path, extracted_file_path):
    """
    Loads cleaned summary document and full extracted text.
    
    Args:
        cleaned_file_path (str): Path to the JSON file containing cleaned results.
        extracted_file_path (str): Path to the file containing full extracted content.

    Returns:
        tuple: (cleaned_results, extracted_text)
    """
    try:
        with open(cleaned_file_path, "r", encoding="utf-8") as f:
            cleaned_data = json.load(f)
            logging.info(f"Loaded cleaned results from {cleaned_file_path}")
        
        with open(extracted_file_path, "r", encoding="utf-8") as f:
            extracted_data = f.read()
            logging.info(f"Loaded extracted text from {extracted_file_path}")
        
        return cleaned_data, extracted_data
    
    except Exception as e:
        logging.error(f"Error loading sources: {str(e)}")
        return None, None

def initialize_globals():
    """
    Initializes global variables for cleaned results and extracted text.
    
    Returns:
        None
    """
    global cleaned_results, extracted_text
    
    # File paths
    cleaned_results_path = "./cleaned_whitepaper_analysis.json"
    extracted_text_path = "./extracted_content/extracted_text.txt"
    
    # Load sources
    cleaned_results, extracted_text = load_sources(cleaned_results_path, extracted_text_path)
    
    if not cleaned_results or not extracted_text:
        logging.error("Failed to load required data files. Please check file paths.")
        raise ValueError("Initialization failed: Missing required data files.")

def process_question(user_query):
    """
    Process a user question by analyzing it and routing it to the appropriate workflow.
    
    Args:
        user_query (str): The user's question.
        
    Returns:
        dict: Answer and metadata appropriate to the query type.
    """
    global cleaned_results, extracted_text
    
    # Ensure globals are initialized
    if not cleaned_results or not extracted_text:
        logging.error("Global variables are not initialized. Call initialize_globals() first.")
        raise ValueError("Global variables are uninitialized.")
    
    # Analyze the query
    analyzed_query = analyze_validation_query(user_query)
    
    # Check if this is a validation-related query
    validation_terms = ["validate", "validation", "verify", "audit", 
                        "assumptions", "accuracy", "metrics", 
                        "performance", "compliance"]
    
    is_validation_query = any(term in analyzed_query["key_terms"] for term in validation_terms)
    
    if is_validation_query:
        logging.info(f"Detected validation query: {analyzed_query['intent']}")
        
        # Pass cleaned_results and extracted_text to answer_validation_query
        return answer_validation_query(user_query, cleaned_results, extracted_text)
    
    logging.info(f"Processing as general question: {analyzed_query['intent']}")
    
    # General queries don't require cleaned_results or extracted_text
    return answer_question(user_query)

def run_qa_cli():
    """
    Run a simple command-line interface for the Q&A system.
    
    Returns:
        None
    """
    try:
        # Initialize global variables
        initialize_globals()
        
        print("\n=== Enhanced Model Validation & Q&A System ===")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            query = input("\nEnter your question: ").strip()
            
            if query.lower() in ['exit', 'quit']:
                print("Exiting Q&A system. Goodbye!")
                break
            
            if not query:
                print("Please enter a valid question.")
                continue
            
            print("\nProcessing your question...\n")
            
            try:
                # Process the question
                result = process_question(query)
                
                # Display results
                print("\n=== ANSWER ===")
                if "technical_confidence" in result:
                    print(f"Technical Confidence: {result.get('technical_confidence', 0)}/5")
                    print(f"Validation Status: {result.get('validation_status', 'Unknown')}")
                else:
                    print(f"Confidence: {result.get('confidence', 0)}/5")
                
                print(result['answer'])
                
                # Display sources
                if "source_sections" in result:
                    print("\n=== SOURCES ===")
                    for source in result["source_sections"]:
                        print(f"- {source}")
                
                # Display follow-up questions
                if result.get("followup_questions"):
                    print("\n=== FOLLOW-UP QUESTIONS ===")
                    for i, q in enumerate(result["followup_questions"], 1):
                        print(f"{i}. {q}")
                
                # Display verification summary for validation queries
                if "verification_summary" in result and result["verification_summary"]:
                    print("\n=== VERIFICATION SUMMARY ===")
                    print(result["verification_summary"])
            
            except Exception as e:
                logging.error(f"Error processing question: {str(e)}")
                import traceback
                print(traceback.format_exc())
            
            print("\n" + "-"*60)
    
    except ValueError as e:
        logging.error(str(e))
        print("Failed to initialize system. Please check log for details.")




























def answer_validation_query(user_query, cleaned_results, extracted_text):
    """
    Full workflow optimized for model validation queries with enhanced accuracy.
    
    Args:
        user_query (str): The model validator's question.
        cleaned_results (dict): Cleaned summary document with structured sections.
        extracted_text (str): Full extracted text of the whitepaper.
        
    Returns:
        dict: Complete response with answer, citations, verification, and confidence.
    """
    # Step 1: Query analysis with technical term detection
    query_dict = analyze_validation_query(user_query)
    
    # Step 2: Enhanced semantic search
    search_results = enhanced_technical_search(query_dict, cleaned_results, extracted_text)
    
    if not search_results:
        return {
            "answer": "I couldn't find relevant technical information to validate this aspect of the model.",
            "confidence": 0,
            "verification_status": "Failed - insufficient information",
            "technical_accuracy": "Unknown - no source material"
        }
    
    # Step 3: Technical context refinement
    refined_context = refine_context_safely(user_query, search_results)
    
    # Step 4: Technical validation answer generation
    detailed_answer = generate_validated_answer(user_query, refined_context)
    
    # Step 5: Technical accuracy verification
    verification = verify_technical_accuracy(user_query, detailed_answer, refined_context)
    
    # Step 6: Apply corrections if critical issues found
    if verification.get("critical_issues", []):
        corrected_answer = apply_technical_corrections(detailed_answer, verification)
        logging.warning(f"Applied corrections to answer: {len(verification.get('critical_issues', []))} critical issues fixed")
    else:
        corrected_answer = detailed_answer
    
    # Step 7: Generate validation-specific follow-up questions
    followup_questions = generate_followup_questions(user_query, corrected_answer)
    
    # Create final response package
    result = {
        "answer": corrected_answer,
        "technical_confidence": calculate_technical_confidence(verification),
        "validation_status": determine_validation_status(verification),
        "source_sections": list(search_results.keys()),
        "verification_summary": verification.get("verification_summary", ""),
        "followup_questions": followup_questions
    }
    
    return result


def process_question(user_query):
    """
    Process a user question by analyzing it and routing it to the appropriate workflow.
    
    Args:
        user_query (str): The user's question.
        
    Returns:
        dict: Answer and metadata appropriate to the query type.
    """
    global cleaned_results, extracted_text
    
    # Analyze the query
    analyzed_query = analyze_validation_query(user_query)
    
    # Check if this is a validation-related query
    validation_terms = ["validate", "validation", "verify", "audit", 
                        "assumptions", "accuracy", "metrics", 
                        "performance", "compliance"]
    
    is_validation_query = any(term in analyzed_query["key_terms"] for term in validation_terms)
    
    if is_validation_query:
        logging.info(f"Detected validation query: {analyzed_query['intent']}")
        
        # Pass cleaned_results and extracted_text to answer_validation_query
        return answer_validation_query(user_query, cleaned_results, extracted_text)
    
    logging.info(f"Processing as general question: {analyzed_query['intent']}")
    
    # General queries don't require cleaned_results or extracted_text
    return answer_question(user_query)































from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

def enhanced_technical_search(query_dict, cleaned_results, extracted_text):
    """
    Performs an enhanced multi-strategy search for technical content.
    
    Args:
        query_dict (dict): The transformed query with intent, key terms, rephrased query, and related queries.
        cleaned_results (dict): Cleaned summary document with structured sections.
        extracted_text (str): Full extracted text of the whitepaper.

    Returns:
        dict: Combined relevant sections from both sources.
    """
    # Extract query components
    rephrased_query = query_dict.get("rephrased_query", "")
    key_terms = query_dict.get("key_terms", [])
    related_queries = query_dict.get("related_queries", [])
    
    # Initialize relevant sections
    relevant_sections = {}
    
    # STEP 1: Search in Cleaned Results (Structured Data)
    logging.info("Searching cleaned results for relevant sections...")
    
    # Define technical section keywords
    section_keywords = {
        "Calculations": {"calculation", "formula", "algorithm", "compute"},
        "Model Performance": {"performance", "accuracy", "precision", "recall", "metrics"},
        "Inputs": {"input", "parameters", "variables", "data"},
        "Outputs": {"output", "results", "predictions"},
        "Testing Summary": {"testing", "test", "validate", "verification"},
        "Solution Specification": {"solution", "specification", "architecture", "design"},
        "Reconciliation": {"reconciliation", "reconcile", "match"}
    }
    
    # Search for relevant sections in cleaned results
    for section_name, terms in section_keywords.items():
        if section_name in cleaned_results:
            # Check if any key term matches the section keywords
            if any(term in key_terms for term in terms):
                logging.info(f"Found relevant section in cleaned results: {section_name}")
                relevant_sections[section_name] = cleaned_results[section_name]
    
    # STEP 2: Semantic Search in Full Extracted Text
    logging.info("Performing semantic search on full extracted text...")
    
    def semantic_search(query, text_chunks, top_k=5):
        """
        Performs semantic search using TF-IDF and cosine similarity.
        
        Args:
            query (str): User's query.
            text_chunks (list): List of text chunks to search through.
            top_k (int): Number of top results to return.

        Returns:
            list: Top matching chunks with similarity scores.
        """
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Fit and transform text chunks
        chunk_vectors = vectorizer.fit_transform(text_chunks)
        
        # Transform query
        query_vector = vectorizer.transform([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, chunk_vectors)[0]
        
        # Get top-k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [(text_chunks[i], similarities[i]) for i in top_indices if similarities[i] > 0.1]
    
    # Split extracted text into paragraphs for semantic search
    paragraphs = [p.strip() for p in extracted_text.split("\n\n") if len(p.strip()) > 50]
    
    # Perform semantic search using rephrased query
    semantic_results = semantic_search(rephrased_query, paragraphs)
    
    # Add top results to relevant sections under 'Additional Context'
    if semantic_results:
        logging.info(f"Found {len(semantic_results)} relevant paragraphs from semantic search.")
        
        additional_context = "\n\n".join([f"Relevance {score:.2f}: {text}" for text, score in semantic_results])
        relevant_sections["Additional Context"] = additional_context
    
    # STEP 3: Expand Search with Related Queries
    logging.info("Expanding search with related queries...")
    
    for related_query in related_queries[:2]:  # Limit to top 2 related queries for efficiency
        related_results = semantic_search(related_query, paragraphs)
        
        if related_results:
            logging.info(f"Found {len(related_results)} additional paragraphs from related query: {related_query}")
            
            additional_related_context = "\n\n".join([f"Relevance {score:.2f}: {text}" for text, score in related_results])
            
            if "Additional Context" not in relevant_sections:
                relevant_sections["Additional Context"] = additional_related_context
            else:
                relevant_sections["Additional Context"] += "\n\n" + additional_related_context
    
    return relevant_sections












import boto3
import json
import logging


def analyze_validation_query(user_query):
    """
    Analyzes a user's validation query to extract intent, key terms, and rephrased versions 
    for optimized semantic search and LLM processing.
    
    Args:
        user_query (str): The original user question or validation request.
        
    Returns:
        dict: A structured dictionary with intent, key terms, rephrased query, and related queries.
    """
    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
    
    # Corrected prompt with properly formatted JSON example
    prompt = f"""
    You are an expert query analyst specializing in model validation tasks. Analyze the following user query:

    USER QUERY: "{user_query}"

    Tasks:
    1. Identify the **main intent** of this query (e.g., "validate performance metrics", "verify assumptions", "audit calculations").
    2. Extract **key search terms** that would be most effective for finding relevant information (3-5 terms).
    3. Generate a **rephrased version** of the query that is optimized for semantic search.
    4. Create a list of **related questions** that might help expand the search scope.

    Instructions:
    - Be concise but precise in identifying the intent.
    - Ensure key terms are specific to model validation (e.g., "accuracy", "precision", "assumptions").
    - Rephrased queries should be clear and unambiguous.
    - Related questions should explore complementary or clarifying aspects of the original query.

    Format your response as JSON with these keys:
    {{
      "intent": "<main_intent>",
      "key_terms": ["<term_1>", "<term_2>", ...],
      "rephrased_query": "<optimized_query>",
      "related_queries": ["<related_question_1>", "<related_question_2>", ...]
    }}
    
    Return only valid JSON output."""
    
    try:
        # Invoke Claude via AWS Bedrock
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
                "temperature": 0.2
            })
        )
        
        # Parse response
        response_body = json.loads(response['body'].read().decode('utf-8'))
        analyzed_query = json.loads(response_body["content"])
        
        logging.info(f"Query analysis completed successfully: {analyzed_query}")
        return analyzed_query
    
    except Exception as e:
        logging.error(f"Error analyzing validation query: {str(e)}")
        
        # Fallback in case of error
        return {
            "intent": user_query,
            "key_terms": user_query.lower().split(),
            "rephrased_query": user_query,
            "related_queries": [user_query]
        }












def apply_technical_corrections(answer, verification):
    """
    Apply corrections to an answer based on verification results.
    
    Args:
        answer (str): The original answer
        verification (dict): Verification results with issues and corrections
        
    Returns:
        str: Answer with corrections applied or marked for review
    """
    # Start with original answer
    corrected = answer
    
    # If there are critical issues that need correction
    if verification.get("critical_issues"):
        # Add warning disclaimer at the top
        correction_warning = """
        [IMPORTANT: This answer contains corrections to technical inaccuracies 
        that were identified during verification. Original content has been 
        modified to ensure technical accuracy.]
        
        """
        
        # We could implement automatic correction here, but for validation
        # it's safer to just highlight issues and let human validators review
        issue_summary = "\n\n== TECHNICAL ACCURACY NOTES ==\n"
        for i, issue in enumerate(verification.get("critical_issues", [])):
            issue_summary += f"\n{i+1}. CRITICAL ISSUE: {issue.get('issue', '')}\n"
            issue_summary += f"   CORRECTION: {issue.get('correction', '')}\n"
            
        for i, issue in enumerate(verification.get("important_issues", [])[:3]):  # Top 3 important issues
            issue_summary += f"\n{i+1}. IMPORTANT ISSUE: {issue.get('issue', '')}\n"
            issue_summary += f"   SUGGESTION: {issue.get('correction', '')}\n"
            
        corrected = correction_warning + corrected + issue_summary
        
    return corrected

def calculate_technical_confidence(verification):
    """
    Calculate technical confidence level based on verification results.
    
    Args:
        verification (dict): Verification results
        
    Returns:
        float: Confidence score from 0-5
    """
    # Default medium confidence
    base_confidence = 3.0
    
    # Adjust based on issues found
    critical_count = len(verification.get("critical_issues", []))
    important_count = len(verification.get("important_issues", []))
    minor_count = len(verification.get("minor_issues", []))
    
    # Critical issues severely impact confidence
    confidence = base_confidence - (critical_count * 1.0)
    
    # Important issues moderately impact confidence
    confidence -= (important_count * 0.3)
    
    # Minor issues slightly impact confidence
    confidence -= (minor_count * 0.1)
    
    # If verification failed completely
    if verification.get("verification_approach") == "error_fallback":
        confidence = min(confidence, 1.0)
    
    # Cap confidence between 0 and 5
    confidence = max(0, min(5, confidence))
    
    return round(confidence, 1)

def determine_validation_status(verification):
    """
    Determine overall validation status based on verification results.
    
    Args:
        verification (dict): Verification results
        
    Returns:
        str: Validation status description
    """
    if verification.get("verification_approach") == "error_fallback":
        return "Indeterminate - verification error"
        
    critical_issues = len(verification.get("critical_issues", []))
    
    if verification.get("verified_accurate", False) and critical_issues == 0:
        return "Validated - No critical issues"
    elif critical_issues > 3:
        return "Failed validation - Multiple critical issues"
    elif critical_issues > 0:
        return "Partially validated - Some critical issues"
    else:
        return "Validated with minor issues"



def process_question(user_query):
    """
    Process a user question, automatically selecting between general questions
    and specialized model validation queries.
    
    Args:
        user_query (str): The user's question
        
    Returns:
        dict: Answer and metadata appropriate to the query type
    """
    # Detect if this is a validation query
    validation_terms = [
        "validate", "validation", "verify", "verification", "audit", 
        "reconcile", "compliance", "accuracy", "precision", "test results",
        "model performance", "benchmark", "regulatory", "requirements"
    ]
    
    is_validation_query = any(term in user_query.lower() for term in validation_terms)
    
    if is_validation_query:
        logging.info(f"Detected validation query: {user_query[:100]}...")
        return answer_validation_query(user_query)
    else:
        logging.info(f"Processing as general query: {user_query[:100]}...")
        return answer_question(user_query)


def run_qa_cli():
    """Run a simple command-line interface for the Q&A system."""
    global cleaned_results, extracted_text, bedrock_client
    
    # File paths
    cleaned_results_path = "./cleaned_whitepaper_analysis.json"
    extracted_text_path = "./extracted_content/extracted_text.txt"
    
    # Load data and set up client
    cleaned_results, extracted_text = load_sources(cleaned_results_path, extracted_text_path)
    bedrock_client = setup_bedrock_client()
    
    # Print status
    print("\n=== Enhanced Model Validation & Q&A System ===")
    print("Type 'exit' or 'quit' to end the session.\n")
    print("This system automatically detects and handles validation queries with enhanced technical accuracy.")
    
    while True:
        query = input("\nEnter your question: ").strip()
        
        if query.lower() in ['exit', 'quit']:
            print("Exiting Q&A system. Goodbye!")
            break
        
        if not query:
            print("Please enter a valid question.")
            continue
        
        print("\nProcessing your question...\n")
        
        try:
            # Process with appropriate workflow
            result = process_question(query)
            
            # Display answer with confidence
            print("\n=== ANSWER ===")
            if "technical_confidence" in result:
                print(f"Technical Confidence: {result.get('technical_confidence', 0)}/5")
                print(f"Validation Status: {result.get('validation_status', 'Unknown')}")
            else:
                print(f"Confidence: {result.get('confidence', 0)}/5")
                
            print(result['answer'])
            
            # Display sources
            if "source_sections" in result:
                print("\n=== SOURCES ===")
                for source in result["source_sections"]:
                    print(f"- {source}")
            elif "sources" in result:
                print("\n=== SOURCES ===")
                for source in result["sources"]:
                    print(f"- {source}")
            
            # Display follow-up questions
            if result.get("followup_questions"):
                print("\n=== FOLLOW-UP QUESTIONS ===")
                for i, q in enumerate(result["followup_questions"], 1):
                    print(f"{i}. {q}")
            
            # Display verification summary for validation queries
            if "verification_summary" in result and result["verification_summary"]:
                print("\n=== VERIFICATION SUMMARY ===")
                print(result["verification_summary"])
            
        except Exception as e:
            print(f"Error processing question: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
        print("\n" + "-"*60)














def verify_technical_accuracy(query, answer, context):
    """
    Verify the technical accuracy of model validation answers by comparing against source context.
    
    Args:
        query (str): The original user query
        answer (str): The generated answer to verify
        context (str): The source context used to generate the answer
        
    Returns:
        dict: Comprehensive verification results with detailed accuracy assessment
    """
    # STEP 1: Initial setup and logging
    logging.info("Beginning technical accuracy verification...")
    
    # STEP 2: Extract key components from the answer
    # Extract numerical values for precise comparison
    numerical_pattern = r'(-?\d+\.?\d*(?:e[+-]?\d+)?)'
    answer_numbers = re.findall(numerical_pattern, answer)
    context_numbers = re.findall(numerical_pattern, context)
    
    # Extract formulas for verification (text between $ signs or anything with = sign)
    formula_pattern = r'\$([^$]+)\$|(\w+\s*=\s*[^.;]+)'
    answer_formulas = re.findall(formula_pattern, answer)
    context_formulas = re.findall(formula_pattern, context)
    
    # Extract section references
    section_references = re.findall(r'==\s*([^=]+?)\s*==', context)
    
    # Check if answer appears to be citing sections
    has_citations = any(section in answer for section in section_references)
    
    # Log initial assessment
    logging.info(f"Found {len(answer_numbers)} numerical values and {len(answer_formulas)} formulas in answer")
    
    # STEP 3: Determine if verification needs complex or simple approach
    needs_complex_verification = len(answer) > 1000 or len(answer_numbers) > 5 or len(answer_formulas) > 2

    # For very small or simple answers, use a simplified verification approach
    if not needs_complex_verification:
        simple_verification_prompt = f"""
        You are an expert technical reviewer verifying the accuracy of a model validation answer.
        
        ORIGINAL QUESTION: {query}
        
        ANSWER TO VERIFY:
        {answer}
        
        SOURCE CONTEXT:
        {context}
        
        Please verify if the answer is technically accurate compared to the source context.
        
        Format your response as a JSON object with these keys:
        "verified_accurate": boolean (true if no critical issues)
        "issues_found": integer count of issues
        "critical_issues": array of critical issues
        "important_issues": array of important issues
        "minor_issues": array of minor issues
        "verification_summary": brief summary of your assessment
        """
        
        try:
            response = bedrock_client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=json.dumps({
                    "messages": [{"role": "user", "content": simple_verification_prompt}],
                    "max_tokens": 1500,
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"}
                })
            )
            
            response_body = json.loads(response['body'].read().decode('utf-8'))
            verification_result = json.loads(response_body["content"])
            verification_result["verification_approach"] = "simple"
            
            logging.info(f"Completed simple verification: {verification_result['issues_found']} issues found")
            return verification_result
            
        except Exception as e:
            logging.error(f"Error in simple verification: {str(e)}")
            return {
                "verified_accurate": False,
                "issues_found": 1,
                "critical_issues": [f"Verification failed due to technical error: {str(e)}"],
                "important_issues": [],
                "minor_issues": [],
                "verification_summary": "Technical error during verification",
                "verification_approach": "error_fallback"
            }
    
    # STEP 4: Prepare context for complex verification
    # If context is too large, extract only the most relevant parts to stay within token limits
    max_context_chars = 80000  # Lower limit for verification to ensure it fits
    
    if len(context) > max_context_chars:
        logging.warning(f"Context too large for verification ({len(context)} chars). Extracting key sections.")
        
        # Approach: Extract sections referenced in the answer, plus high-value sections
        extracted_context = "== Extracted Relevant Sections =="
        context_sections = re.split(r'(==\s.*?\s==)', context)
        
        # Track what we've added and total size
        added_sections = []
        current_size = len(extracted_context)
        
        # First, extract sections that are explicitly referenced in the answer
        for i in range(1, len(context_sections) - 1, 2):  # Process section headers
            if i+1 < len(context_sections):
                section_header = context_sections[i]
                section_content = context_sections[i+1]
                
                # Check if this section is referenced in the answer
                section_name = section_header.strip("= \n")
                if section_name in answer:
                    section_text = section_header + section_content
                    # Check if adding would exceed limit
                    if current_size + len(section_text) < max_context_chars:
                        extracted_context += "\n\n" + section_text
                        current_size += len(section_text)
                        added_sections.append(section_name)
                        logging.info(f"Added referenced section to verification context: {section_name}")
        
        # Next, add critical sections for validation if not already added
        critical_sections = ["Calculations", "Model Performance", "Testing", "Inputs", "Outputs"]
        for i in range(1, len(context_sections) - 1, 2):
            if i+1 < len(context_sections):
                section_header = context_sections[i]
                section_content = context_sections[i+1]
                section_name = section_header.strip("= \n")
                
                if any(critical in section_name for critical in critical_sections) and section_name not in added_sections:
                    section_text = section_header + section_content
                    # Check if adding would exceed limit
                    if current_size + len(section_text) < max_context_chars:
                        extracted_context += "\n\n" + section_text
                        current_size += len(section_text)
                        added_sections.append(section_name)
                        logging.info(f"Added critical section to verification context: {section_name}")
        
        # Use the extracted context for verification
        verification_context = extracted_context
        logging.info(f"Created verification context with {len(added_sections)} sections, {current_size} chars")
    else:
        verification_context = context
    
    # STEP 5: Create verification prompt optimized for technical validation
    verification_prompt = f"""
    You are an expert technical reviewer verifying the accuracy of a model validation answer.
    
    ORIGINAL QUESTION: {query}
    
    ANSWER TO VERIFY:
    {answer}
    
    SOURCE CONTEXT:
    {verification_context}
    
    Perform a thorough technical accuracy review focusing on these aspects:
    
    1. NUMERICAL ACCURACY: Verify all numerical values match the source context
    2. FORMULA ACCURACY: Verify all formulas and equations match the source
    3. METHODOLOGICAL ACCURACY: Verify descriptions of methods/processes are accurate
    4. CITATION ACCURACY: Verify references to sections or sources are correct
    5. COMPLETENESS: Verify that important context information was not omitted
    
    For each issue found, classify it by severity:
    - CRITICAL: Factually incorrect numbers, formulas, or methodological claims that would mislead
    - IMPORTANT: Significant omissions, unclear explanations, or imprecise descriptions
    - MINOR: Style issues, minor imprecisions, or areas that could be clarified
    
    For each issue, provide:
    - Specific text from the answer that contains the issue
    - The correct information from the context
    - Clear correction suggestion
    
    Format your response as a JSON object with these keys:
    "verified_accurate": boolean (true if no critical issues)
    "issues_found": integer count of issues
    "critical_issues": array of critical issues, each with "issue", "correction", and "location"
    "important_issues": array of important issues, each with "issue", "correction", and "location"
    "minor_issues": array of minor issues, each with "issue", "correction", and "location"
    "verification_summary": descriptive assessment summarizing your findings
    "suggested_corrections": specific text suggestions to improve accuracy
    """
    
    # STEP 6: Call the model for verification
    try:
        logging.info("Sending verification request to Claude...")
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": verification_prompt}],
                "max_tokens": 2500,
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        verification_result = json.loads(response_body["content"])
        
        # STEP 7: Enhance results with additional metadata
        verification_result["verification_approach"] = "comprehensive"
        
        # Add supplementary accuracy metrics
        verification_result["accuracy_metrics"] = {
            "numerical_values_count": len(answer_numbers),
            "formulas_count": len(answer_formulas),
            "has_citations": has_citations,
            "critical_issue_rate": len(verification_result.get("critical_issues", [])) / max(1, len(answer_numbers) + len(answer_formulas)),
            "verification_confidence": "high" if len(verification_context) > 10000 else "medium"
        }
        
        # Log verification results
        critical_count = len(verification_result.get("critical_issues", []))
        important_count = len(verification_result.get("important_issues", []))
        minor_count = len(verification_result.get("minor_issues", []))
        
        logging.info(f"Verification complete: {critical_count} critical, {important_count} important, {minor_count} minor issues")
        
        return verification_result
        
    except Exception as e:
        logging.error(f"Error in technical verification: {str(e)}")
        
        # STEP 8: Fallback error handling
        return {
            "verified_accurate": False,
            "issues_found": 1,
            "critical_issues": [{"issue": f"Verification failed due to technical error: {str(e)}", 
                                "correction": "Manual verification required", 
                                "location": "entire document"}],
            "important_issues": [],
            "minor_issues": [],
            "verification_summary": "Technical error during verification process prevented detailed analysis",
            "verification_approach": "error_fallback"
        }




def refine_context_safely(query, raw_context):
    """
    Enhanced context management for model validation use cases.
    Prioritizes technical content and preserves precise details critical for validation.
    
    Args:
        query (str): The user's question or validation request
        raw_context (dict): The raw search results containing sections of content
        
    Returns:
        str: Refined and optimized context with technical precision
    """
    try:
        # STEP 1: Determine max safe context size (increased for model validation needs)
        max_context_chars = 100000  # ~25K tokens, still conservative but more flexible for technical content
        
        # STEP 2: Detect technical validation queries
        technical_terms = [
            "validation", "accuracy", "precision", "recall", "f1", "error rate", 
            "confidence interval", "statistical", "margin of error", "test set", 
            "training data", "evaluation metrics", "performance", "calculation",
            "methodology", "formula", "algorithm", "technical", "specification",
            "reconciliation", "verification", "audit", "compliance", "testing"
        ]
        
        is_technical_query = any(term in query.lower() for term in technical_terms)
        logging.info(f"Query classified as {'technical validation' if is_technical_query else 'general'} query")
        
        # STEP 3: Prepare priority order based on query type
        if is_technical_query:
            # Technical validation prioritizes calculations and performance metrics
            priority_sections = [
                "Calculations",       # Highest priority for model validation
                "Model Performance",  # Critical for validation
                "Inputs",             # Essential for understanding model
                "Outputs",            # Essential for understanding model
                "Testing Summary",    # Important for validation methodologies
                "Solution Specification", # Technical details
                "Reconciliation",     # Validation processes
                "Summary",            # General context
                "Additional Context"  # Supplementary information
            ]
            logging.info("Using technical validation section prioritization")
        else:
            # General queries use default priority
            priority_sections = [
                "Summary",           # High priority - overall understanding
                "Additional Context", # High priority - search results
                "Inputs",            # Medium priority
                "Outputs",           # Medium priority
                "Calculations",      # Medium priority
                "Model Performance", # Lower priority
                "Solution Specification", # Lower priority
                "Testing Summary",   # Lower priority
                "Reconciliation"     # Lower priority
            ]
            logging.info("Using general query section prioritization")
        
        # STEP 4: Format sections in priority order
        formatted_sections = []
        current_length = 0
        added_sections = []
        skipped_sections = []
        
        # Track what was processed
        logging.info(f"Beginning context construction with {len(raw_context)} available sections")
        
        # STEP 5: Add sections in priority order
        for section_name in priority_sections:
            if section_name in raw_context:
                try:
                    # Extract and format content
                    content = raw_context[section_name]
                    section_text = f"== {section_name} ==\n{format_content(content)}"
                    section_length = len(section_text)
                    
                    logging.info(f"Section '{section_name}' has {section_length} characters")
                    
                    # Check space with technical prioritization logic
                    if current_length + section_length > max_context_chars:
                        # For technical validation, prioritize technical sections even if truncation needed
                        if is_technical_query and section_name in ["Calculations", "Model Performance", "Testing Summary"]:
                            # For critical technical sections, include even if we need to truncate
                            available_space = max_context_chars - current_length - 100  # 100 char buffer
                            if available_space <= 0:
                                logging.warning(f"No space left for {section_name}")
                                skipped_sections.append(section_name)
                                continue
                                
                            truncated_text = section_text[:available_space]
                            truncated_text += "\n[Note: Some technical content was truncated due to length constraints]"
                            formatted_sections.append(truncated_text)
                            added_sections.append(section_name)
                            current_length += len(truncated_text)
                            logging.warning(f"Added truncated {section_name}: {len(truncated_text)} chars")
                        
                        # For non-technical queries or non-critical sections, handle normally    
                        elif section_name in ["Summary", "Additional Context"] and section_name not in added_sections:
                            available_space = max_context_chars - current_length - 100
                            if available_space <= 0:
                                logging.warning(f"No space left for {section_name}")
                                skipped_sections.append(section_name)
                                continue
                                
                            truncated_text = section_text[:available_space]
                            truncated_text += "\n[Content truncated due to length]"
                            formatted_sections.append(truncated_text)
                            added_sections.append(section_name)
                            current_length += len(truncated_text)
                            logging.warning(f"Added truncated {section_name}: {len(truncated_text)} chars")
                        else:
                            logging.info(f"Skipping {section_name}: would exceed limit")
                            skipped_sections.append(section_name)
                    else:
                        # Section fits, add it normally
                        formatted_sections.append(section_text)
                        added_sections.append(section_name)
                        current_length += section_length
                        logging.info(f"Added {section_name}: {section_length} chars, total now: {current_length}")
                except Exception as section_error:
                    logging.error(f"Error processing section {section_name}: {str(section_error)}")
                    skipped_sections.append(section_name)
                    # Continue with other sections
        
        # STEP 6: Emergency handling for no sections
        if not formatted_sections:
            logging.warning("No sections could fit. Creating minimal context.")
            first_section_name = next(iter(raw_context.keys())) if raw_context else "No Data"
            first_content = raw_context[first_section_name] if raw_context else "No content available"
            
            # Create minimal context with first section, severely truncated
            minimal_text = f"== {first_section_name} ==\n"
            available_space = max_context_chars - len(minimal_text) - 50
            minimal_text += format_content(first_content)[:available_space]
            minimal_text += "\n[Severely truncated due to length constraints]"
            
            formatted_sections = [minimal_text]
            added_sections.append(first_section_name)
        
        # STEP 7: Join sections to form context
        context_text = "\n\n".join(formatted_sections)
        logging.info(f"Final context size: {len(context_text)} characters, included {len(added_sections)} sections")
        if skipped_sections:
            logging.info(f"Skipped sections: {', '.join(skipped_sections)}")
        
        # STEP 8: Return context without refinement if too large or tiny
        if len(context_text) > (max_context_chars * 0.8):
            logging.info("Context too large for refinement step. Returning as is.")
            return context_text
        
        if len(context_text) < 1000:
            logging.info("Context too small for refinement. Returning as is.")
            return context_text
            
        # STEP 9: Prepare refinement prompt based on query type
        logging.info("Attempting context refinement...")
        
        if is_technical_query:
            # Technical validation prompt with emphasis on precision and accuracy
            refinement_prompt = f"""
            You are an expert model validator analyzing technical documentation.
            
            USER QUERY: {query}
            
            CONTEXT:
            {context_text}
            
            Please optimize this context for answering the technical validation query:
            1. Preserve ALL numerical values, formulas, metrics, and technical specifications exactly
            2. Maintain all test results and performance metrics with their precise values
            3. Keep methodological details intact
            4. Organize information by relevance to the validation question
            5. Preserve all section headings (== Section Name ==)
            
            Technical accuracy is CRITICAL. Never round numbers or simplify technical details.
            Return only the refined context, maintaining all section markers and exact values.
            """
        else:
            # Standard refinement prompt for general queries
            refinement_prompt = f"""
            You are an expert information analyst. I have a user question and some context.
            Please analyze this context and optimize it for answering the question.
            
            USER QUESTION: {query}
            
            CONTEXT:
            {context_text}
            
            Please:
            1. Remove any irrelevant portions that don't help answer the question
            2. Reorganize the remaining information in order of relevance to the question
            3. Preserve all section headings (== Section Name ==) and their structure
            4. Preserve all relevant data points, numbers, and specific details
            
            Return only the refined context, maintaining the section markers and formatting.
            """
        
        # STEP 10: Check if prompt is safe to send
        if len(refinement_prompt) > max_context_chars:
            logging.warning("Prompt too large for refinement. Returning original context.")
            return context_text
        
        # STEP 11: Send to Claude with strict timeout and error handling
        try:
            response = bedrock_client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=json.dumps({
                    "messages": [{"role": "user", "content": refinement_prompt}],
                    "max_tokens": 4000,
                    "temperature": 0.2
                })
            )
            
            response_body = json.loads(response['body'].read().decode('utf-8'))
            refined_context = response_body["content"].strip()
            
            # STEP 12: Validate refinement result
            if "==" not in refined_context:
                logging.warning("Refinement removed section markers. Using original context.")
                return context_text
                
            # Additional validation for technical queries
            if is_technical_query:
                # Verify key technical terms were preserved
                original_numbers = re.findall(r'\b\d+\.?\d*\b', context_text)
                refined_numbers = re.findall(r'\b\d+\.?\d*\b', refined_context)
                
                # Check if most important numbers were preserved
                if len(original_numbers) > 0 and len(refined_numbers) < len(original_numbers) * 0.7:
                    logging.warning("Refinement lost significant numerical data. Using original context.")
                    return context_text
            
            logging.info(f"Refinement complete. New size: {len(refined_context)} characters")
            return refined_context
            
        except Exception as refine_error:
            logging.error(f"Error during refinement: {str(refine_error)}")
            return context_text
            
    except Exception as e:
        logging.error(f"Critical error in context refinement: {str(e)}")
        # Return a minimal safe context in case of complete failure
        if raw_context and isinstance(raw_context, dict) and len(raw_context) > 0:
            # Try to extract at least something from raw_context
            first_section = next(iter(raw_context.items()))
            return f"== {first_section[0]} ==\n{format_content(first_section[1])[:5000]}\n\n[Error: Full context processing failed]"
        else:
            return "== Error ==\nUnable to process context properly due to technical issues."





def generate_validated_answer(query, context):
    """
    Generate answers with enhanced accuracy checks specifically for model validation.
    Emphasizes technical precision, uncertainty quantification, and methodological assessment.
    
    Args:
        query (str): The user's question or validation request
        context (str): The refined context containing relevant information
        
    Returns:
        str: A detailed, technically accurate answer with structured reasoning
    """
    # STEP 1: Initial setup and logging
    logging.info(f"Generating validated answer for query: {query[:100]}...")
    
    # Detect technical validation aspects
    validation_aspects = {
        "methodology": any(term in query.lower() for term in ["methodology", "process", "approach", "procedure"]),
        "performance": any(term in query.lower() for term in ["performance", "accuracy", "precision", "recall", "metrics"]),
        "compliance": any(term in query.lower() for term in ["compliance", "regulatory", "requirement", "standard"]),
        "calculations": any(term in query.lower() for term in ["calculation", "formula", "compute", "algorithm"]),
        "testing": any(term in query.lower() for term in ["test", "validate", "verification", "evaluation"]),
        "reconciliation": any(term in query.lower() for term in ["reconcile", "match", "compare", "difference"])
    }
    
    # Log detected aspects
    detected_aspects = [aspect for aspect, detected in validation_aspects.items() if detected]
    if detected_aspects:
        logging.info(f"Detected validation aspects: {', '.join(detected_aspects)}")
    else:
        logging.info("No specific validation aspects detected, using general validation approach")
    
    # STEP 2: Check context length and truncate if needed
    max_context_chars = 100000  # ~25K tokens (safe limit for Claude)
    
    if len(context) > max_context_chars:
        logging.warning(f"Context too large for answer generation ({len(context)} chars). Truncating.")
        # Smarter truncation preserving section headers
        sections = re.split(r'(==\s.*?\s==)', context)
        
        truncated_context = sections[0] if sections[0] != '' else ''  # Start with any text before first header
        current_length = len(truncated_context)
        
        # Add sections until we approach the limit
        for i in range(1, len(sections) - 1, 2):  # Sections come in pairs (header, content)
            if i+1 < len(sections):
                section_header = sections[i]
                section_content = sections[i+1]
                section_length = len(section_header) + len(section_content)
                
                if current_length + section_length > max_context_chars:
                    # If this is a critical section, truncate instead of skipping
                    critical_section = any(term in section_header.lower() for term in 
                                          ["calculation", "performance", "testing", "input", "output"])
                    
                    if critical_section and detected_aspects:
                        available_space = max_context_chars - current_length - len(section_header) - 100
                        if available_space > 200:  # Only if reasonable space remains
                            truncated_section = section_content[:available_space] + "\n[Truncated]"
                            truncated_context += section_header + truncated_section
                            current_length += len(section_header) + len(truncated_section)
                            logging.info(f"Added truncated critical section: {section_header.strip()}")
                    else:
                        logging.info(f"Skipping section: {section_header.strip()}")
                else:
                    truncated_context += section_header + section_content
                    current_length += section_length
                    logging.info(f"Added section: {section_header.strip()}")
        
        context = truncated_context
        context += "\n\n[Note: Some context was truncated due to length constraints]"
    
    # STEP 3: Select appropriate prompt based on validation aspects
    if validation_aspects["methodology"]:
        # Methodology-focused prompt
        prompt_template = """
        You are an expert model validator evaluating a financial model's methodology.
        
        CONTEXT:
        {context}
        
        VALIDATION QUESTION: {query}
        
        Please assess this methodology with this structured approach:
        
        1. ANALYSIS: Identify the specific methodological elements that need validation
        2. DOCUMENTATION: Extract and evaluate the documented methodology steps
        3. STANDARDS ASSESSMENT: Assess adherence to industry standards and best practices
        4. GAP ANALYSIS: Identify any missing or incomplete methodological elements 
        5. UNCERTAINTY EVALUATION: Assess which aspects have high/medium/low documentation quality
        6. CONCLUSION: Provide your methodological assessment with clear confidence levels
        
        CRITICAL GUIDELINES:
        - Assign explicit confidence levels (High/Medium/Low) to each methodological assertion
        - Cite specific sections from the context for methodological details
        - When methodology details are missing, explicitly state what additional documentation is needed
        - Distinguish between documented facts and your professional validation judgment
        - Be precise about methodological strengths and weaknesses
        
        Begin with "ANALYSIS:" and conclude with "CONCLUSION:".
        """
    elif validation_aspects["performance"]:
        # Performance-focused prompt
        prompt_template = """
        You are an expert model validator evaluating a financial model's performance metrics.
        
        CONTEXT:
        {context}
        
        VALIDATION QUESTION: {query}
        
        Please assess this model's performance with this structured approach:
        
        1. ANALYSIS: Identify the key performance metrics and benchmarks that need evaluation
        2. METRICS EXTRACTION: Extract all performance metrics with their exact values
        3. BENCHMARKING: Compare metrics against any stated benchmarks or industry standards
        4. TESTING ASSESSMENT: Evaluate the testing methodology used to derive these metrics
        5. STATISTICAL ANALYSIS: Assess statistical significance and confidence intervals
        6. CONCLUSION: Provide your performance assessment with clear confidence levels
        
        CRITICAL GUIDELINES:
        - Maintain ALL numerical values with their original precision (never round)
        - Include units of measurement for all metrics
        - Cite specific sections from the context for each performance claim
        - Assign a confidence level (High/Medium/Low) to your assessment of each metric
        - When performance data is missing, explicitly state what additional metrics are needed
        - Evaluate testing sample sizes and methodological soundness where mentioned
        
        Begin with "ANALYSIS:" and conclude with "CONCLUSION:".
        """
    elif validation_aspects["calculations"]:
        # Calculations-focused prompt
        prompt_template = """
        You are an expert model validator evaluating a financial model's calculations and algorithms.
        
        CONTEXT:
        {context}
        
        VALIDATION QUESTION: {query}
        
        Please assess this model's calculations with this structured approach:
        
        1. ANALYSIS: Identify the specific calculations and formulas that need validation
        2. FORMULA EXTRACTION: Extract all relevant formulas and calculations exactly as stated
        3. VARIABLES ASSESSMENT: Identify all variables used and their definitions
        4. ALGORITHMIC LOGIC: Evaluate the logical flow and computational steps
        5. EDGE CASE HANDLING: Assess how the calculations handle boundary conditions
        6. CONCLUSION: Provide your calculation assessment with clear confidence levels
        
        CRITICAL GUIDELINES:
        - Preserve ALL mathematical formulas exactly as written
        - Maintain precise notation including subscripts, Greek letters, etc.
        - Keep ALL numerical values with their exact precision
        - Cite specific sections from the context for each calculation
        - Assign a confidence level (High/Medium/Low) to your assessment of each calculation
        - When calculations are unclear, explicitly identify ambiguities
        - Distinguish between documented calculations and your inferences
        
        Begin with "ANALYSIS:" and conclude with "CONCLUSION:".
        """
    else:
        # General validation prompt
        prompt_template = """
        You are an expert model validator providing technical assessment of a financial model.
        
        CONTEXT:
        {context}
        
        VALIDATION QUESTION: {query}
        
        Please follow this structured analytical process:
        
        1. ANALYSIS: Identify exactly what technical aspects need to be validated
        2. EVIDENCE: Extract precise values, formulas, and methodologies from the context
        3. VERIFICATION: Check for internal consistency and methodological soundness
        4. REASONING: Apply validation principles to assess the model's technical merits
        5. UNCERTAINTY: Explicitly note any areas where information is incomplete
        6. CONCLUSION: Provide your technical assessment with appropriate confidence levels
        
        CRITICAL GUIDELINES:
        - Cite specific sections, numbers, and formulas with exact precision
        - Never round numbers or simplify technical details
        - Explicitly state confidence levels for each technical assertion (High/Medium/Low)
        - Clearly distinguish between facts from the context and your professional judgment
        - When information is missing, explicitly state what additional data would be needed
        
        Begin with "ANALYSIS:" and conclude with "CONCLUSION:".
        """
    
    # STEP 4: Format prompt
    prompt = prompt_template.format(query=query, context=context)
    
    # STEP 5: Final token limit check
    # If prompt is still too big for some reason, reduce context further
    if len(prompt) > max_context_chars:
        # Recalculate how much context we can include
        template_size = len(prompt_template.format(query=query, context=""))
        available_size = max_context_chars - template_size - 100  # 100 char buffer
        
        # Truncate context to fit
        context_truncated = context[:available_size]
        logging.warning(f"Further truncated context to {len(context_truncated)} chars to fit prompt template")
        
        # Regenerate prompt
        prompt = prompt_template.format(query=query, context=context_truncated)
    
    # STEP 6: Call model with enhanced error handling
    try:
        logging.info("Sending validation query to Claude...")
        
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000,
                "temperature": 0.3  # Lower temperature for higher precision
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        detailed_answer = response_body["content"].strip()
        
        # STEP 7: Validate answer structure
        expected_sections = ["ANALYSIS", "EVIDENCE", "VERIFICATION", "REASONING", "CONCLUSION"]
        
        # Check for at least some of the expected structure
        has_valid_structure = any(section in detailed_answer for section in expected_sections)
        
        if not has_valid_structure:
            logging.warning("Answer lacks expected validation structure. Adding structural elements.")
            
            # Add minimal structure to unstructured response
            enhanced_answer = f"""
            ANALYSIS:
            Based on the validation question: "{query}"
            
            {detailed_answer}
            
            CONCLUSION:
            The above represents the available information from the documentation.
            """
            return enhanced_answer
        
        # STEP 8: Check for confidence levels
        confidence_terms = ["high confidence", "medium confidence", "low confidence", 
                           "high certainty", "medium certainty", "low certainty"]
        
        has_confidence = any(term in detailed_answer.lower() for term in confidence_terms)
        
        if not has_confidence and "CONCLUSION" in detailed_answer:
            # Split by sections to find conclusion
            sections = re.split(r'(ANALYSIS:|EVIDENCE:|VERIFICATION:|REASONING:|UNCERTAINTY:|CONCLUSION:)', detailed_answer)
            conclusion_idx = None
            
            for i, section in enumerate(sections):
                if section == "CONCLUSION:":
                    conclusion_idx = i
                    break
            
            if conclusion_idx is not None and conclusion_idx + 1 < len(sections):
                # Add confidence disclaimer to conclusion
                conclusion = sections[conclusion_idx + 1]
                enhanced_conclusion = conclusion + "\n\nNOTE: Confidence levels were not explicitly provided in the available documentation. Professional judgment should be exercised when interpreting these results."
                sections[conclusion_idx + 1] = enhanced_conclusion
                detailed_answer = "".join(sections)
                logging.info("Added confidence level disclaimer to conclusion")
        
        logging.info("Successfully generated validated answer")
        return detailed_answer
        
    except Exception as e:
        logging.error(f"Error generating detailed answer: {str(e)}")
        
        # STEP 9: Emergency fallback answer
        fallback_answer = f"""
        ANALYSIS:
        I attempted to analyze the validation question: "{query}"
        
        TECHNICAL DIFFICULTY:
        I encountered a technical issue while generating a comprehensive validation response.
        
        AVAILABLE INFORMATION:
        The context contains information about: {", ".join([s for s in context.split("==") if s.strip() and not s.strip().endswith("==")])}
        
        LIMITED CONCLUSION:
        Due to technical limitations, I cannot provide a complete validation assessment at this time. 
        The raw context should be reviewed manually by a validation expert.
        
        CONFIDENCE: Low (due to technical processing limitations)
        """
        
        return fallback_answer



def verify_technical_accuracy(query, answer, context):
    """Verify the technical accuracy of model validation answers."""
    verification_prompt = f"""
    You are an expert technical reviewer verifying the accuracy of a model validation answer.
    
    ORIGINAL QUESTION: {query}
    
    ANSWER TO VERIFY:
    {answer}
    
    SOURCE CONTEXT:
    {context}
    
    Perform a thorough technical accuracy review:
    
    1. Check all numerical values against the source context
    2. Verify that formulas and methodologies are correctly described
    3. Confirm that technical terminology is used appropriately
    4. Ensure that limitations and assumptions are properly stated
    5. Verify that confidence levels are appropriate given the evidence
    
    For each issue found, provide:
    - The specific inaccuracy
    - The correct information from the context
    - The severity (Critical/Important/Minor)
    
    Format your response as a JSON object with these keys:
    "verified_accurate": boolean (true if no critical issues)
    "issues_found": integer count of issues
    "critical_issues": array of critical issues
    "important_issues": array of important issues
    "minor_issues": array of minor issues
    "suggested_corrections": specific corrections for each issue
    """
    
    # Implement accuracy verification with Claude call
    # ...

def answer_validation_query(user_query):
    """
    Full workflow optimized for model validation queries with enhanced accuracy.
    
    Args:
        user_query (str): The model validator's question
        
    Returns:
        dict: Complete response with answer, citations, verification, and confidence
    """
    # Step 1: Query analysis with technical term detection
    query_dict = analyze_validation_query(user_query)  # Enhanced version
    
    # Step 2: Enhanced semantic search with technical focus
    search_results = enhanced_technical_search(query_dict)  # Enhanced version
    
    if not search_results:
        return {
            "answer": "I couldn't find relevant technical information to validate this aspect of the model.",
            "confidence": 0,
            "verification_status": "Failed - insufficient information",
            "technical_accuracy": "Unknown - no source material"
        }
    
    # Step 3: Technical context refinement
    refined_context = refine_context_safely(user_query, search_results)
    
    # Step 4: Technical validation answer generation
    detailed_answer = generate_validated_answer(user_query, refined_context)
    
    # Step 5: Add precise source citations
    cited_answer = format_answer_with_citations(detailed_answer, search_results)
    
    # Step 6: Technical accuracy verification
    verification = verify_technical_accuracy(user_query, cited_answer, refined_context)
    
    # Step 7: Correct critical issues if any found
    if verification.get("critical_issues"):
        corrected_answer = apply_technical_corrections(cited_answer, verification)
    else:
        corrected_answer = cited_answer
    
    # Step 8: Generate validation-specific follow-up questions
    followup_questions = generate_validation_followups(user_query, corrected_answer)
    
    # Create final response package
    result = {
        "answer": corrected_answer,
        "technical_confidence": calculate_technical_confidence(verification),
        "validation_status": determine_validation_status(verification),
        "source_sections": list(search_results.keys()),
        "verification_summary": verification.get("verification_summary", ""),
        "followup_questions": followup_questions
    }
    
    return result
























def refine_context_safely(query, raw_context):
    """
    A more robust implementation of context refinement that breaks the process
    into small steps with explicit error handling.
    
    Args:
        query (str): The user's question
        raw_context (dict): The raw search results
        
    Returns:
        str: Refined and optimized context
    """
    try:
        # STEP 1: Determine max safe context size
        max_context_chars = 48000  # Ultra-conservative limit (approx 12K tokens)
        
        # STEP 2: Prepare priority order
        priority_sections = [
            "Summary", "Additional Context", "Inputs", "Outputs", 
            "Calculations", "Model Performance", "Solution Specification", 
            "Testing Summary", "Reconciliation"
        ]
        
        # STEP 3: Format sections in priority order
        formatted_sections = []
        current_length = 0
        added_sections = []
        
        logging.info(f"Beginning context construction with {len(raw_context)} available sections")
        
        # STEP 4: Add sections in priority order
        for section_name in priority_sections:
            if section_name in raw_context:
                try:
                    # Extract and format content
                    content = raw_context[section_name]
                    section_text = f"== {section_name} ==\n{format_content(content)}"
                    section_length = len(section_text)
                    
                    logging.info(f"Section '{section_name}' has {section_length} characters")
                    
                    # Check space
                    if current_length + section_length > max_context_chars:
                        # Handle high-priority sections differently
                        if section_name in ["Summary", "Additional Context"] and section_name not in added_sections:
                            available_space = max_context_chars - current_length - 100
                            if available_space <= 0:
                                logging.warning(f"No space left for {section_name}")
                                continue
                                
                            truncated_text = section_text[:available_space] + "\n[Truncated]"
                            formatted_sections.append(truncated_text)
                            added_sections.append(section_name)
                            current_length += len(truncated_text)
                            logging.warning(f"Added truncated {section_name}: {len(truncated_text)} chars")
                        else:
                            logging.info(f"Skipping {section_name}: would exceed limit")
                    else:
                        # Section fits, add it
                        formatted_sections.append(section_text)
                        added_sections.append(section_name)
                        current_length += section_length
                        logging.info(f"Added {section_name}: {section_length} chars, total now: {current_length}")
                except Exception as section_error:
                    logging.error(f"Error processing section {section_name}: {str(section_error)}")
                    # Continue with other sections
        
        # STEP 5: Emergency handling for no sections
        if not formatted_sections:
            logging.warning("No sections could fit. Creating minimal context.")
            formatted_sections = ["== Minimal Context ==\nNo complete sections could fit within limits."]
        
        # STEP 6: Join sections to form context
        context_text = "\n\n".join(formatted_sections)
        logging.info(f"Final context size: {len(context_text)} characters")
        
        # STEP 7: Return context without refinement if too large
        if len(context_text) > (max_context_chars * 0.8):
            logging.info("Context too large for refinement step. Returning as is.")
            return context_text
        
        # STEP 8: Skip refinement for very small contexts
        if len(context_text) < 1000:
            logging.info("Context too small for refinement. Returning as is.")
            return context_text
            
        # STEP 9: Try to refine context
        logging.info("Attempting context refinement...")
        prompt = f"""
        You are an expert analyst. Optimize this context for answering: "{query}"
        
        CONTEXT:
        {context_text}
        
        Instructions:
        1. Remove irrelevant information
        2. Preserve section headings (== Section Name ==)
        3. Focus on information most relevant to the question
        """
        
        # STEP 10: Check if prompt is safe to send
        if len(prompt) > max_context_chars:
            logging.warning("Prompt too large for refinement. Returning original context.")
            return context_text
        
        # STEP 11: Send to Claude with strict timeout
        try:
            response = bedrock_client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=json.dumps({
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4000,
                    "temperature": 0.2
                })
            )
            
            response_body = json.loads(response['body'].read().decode('utf-8'))
            refined_context = response_body["content"].strip()
            
            # STEP 12: Validate refinement result
            if "==" not in refined_context:
                logging.warning("Refinement removed section markers. Using original context.")
                return context_text
                
            logging.info(f"Refinement complete. New size: {len(refined_context)} characters")
            return refined_context
            
        except Exception as refine_error:
            logging.error(f"Error during refinement: {str(refine_error)}")
            return context_text
            
    except Exception as e:
        logging.error(f"Critical error in context refinement: {str(e)}")
        # Return a minimal safe context in case of complete failure
        return "== Error ==\nUnable to process context properly due to technical issues."







import os
import json
import logging
import boto3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables to store our data
cleaned_results = None
extracted_text = None
bedrock_client = None
answer_cache = {}

def load_sources(cleaned_file_path, extracted_file_path):
    """
    Load source data from files.
    
    Args:
        cleaned_file_path (str): Path to the JSON file containing cleaned results.
        extracted_file_path (str): Path to the file containing full extracted content.
    
    Returns:
        tuple: (cleaned_results, extracted_text)
    """
    cleaned_data, extracted_data = None, None
    
    try:
        if os.path.exists(cleaned_file_path):
            with open(cleaned_file_path, "r", encoding="utf-8") as f:
                cleaned_data = json.load(f)
            logging.info(f"Loaded cleaned results from {cleaned_file_path}")
        else:
            logging.warning(f"Cleaned results file not found: {cleaned_file_path}")
        
        if os.path.exists(extracted_file_path):
            with open(extracted_file_path, "r", encoding="utf-8") as f:
                extracted_data = f.read()
            logging.info(f"Loaded extracted text from {extracted_file_path}")
        else:
            logging.warning(f"Extracted text file not found: {extracted_file_path}")
        
        return cleaned_data, extracted_data
    except Exception as e:
        logging.error(f"Error loading sources: {str(e)}")
        return None, None

def setup_bedrock_client():
    """Set up the AWS Bedrock client."""
    return boto3.client('bedrock-runtime', region_name='us-east-1')

def analyze_and_transform_query(user_query):
    """
    Use LLM to analyze the query and transform it into optimal search queries.
    
    Args:
        user_query (str): The original user question
        
    Returns:
        dict: Various forms of the query for different search strategies
    """
    prompt = f"""
    You are an expert search query analyst. Analyze this user question and help optimize it for search:
    
    Original question: "{user_query}"
    
    1. Identify the main intent of this question.
    2. Extract 3-5 key search terms that would be most effective for finding relevant information.
    3. Generate a rephrased version of this question that would maximize semantic search effectiveness.
    4. Create a list of related questions that might help expand the search scope.
    
    Format your response as JSON with these keys:
    "intent", "key_terms", "rephrased_query", "related_queries"
    """
    
    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        transformed_query = json.loads(response_body["content"])
        
        logging.info(f"Transformed query: {transformed_query}")
        return transformed_query
    except Exception as e:
        logging.error(f"Error analyzing query: {str(e)}")
        # Fallback to simpler transformation
        return {
            "intent": user_query,
            "key_terms": user_query.lower().split(),
            "rephrased_query": user_query,
            "related_queries": [user_query]
        }

def semantic_search(query, text_chunks, top_k=5):
    """
    Performs semantic search using TF-IDF and cosine similarity.
    
    Args:
        query (str): User's query.
        text_chunks (list): List of text chunks to search through.
        top_k (int): Number of top results to return.

    Returns:
        list: Top matching chunks with similarity scores.
    """
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Filter out empty chunks
    valid_chunks = [chunk for chunk in text_chunks if chunk and len(chunk.strip()) > 50]
    
    if not valid_chunks:
        return []
    
    # Fit and transform text chunks
    try:
        chunk_vectors = vectorizer.fit_transform(valid_chunks)
        
        # Transform query
        query_vector = vectorizer.transform([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, chunk_vectors)[0]
        
        # Get top-k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return top chunks with scores
        return [(valid_chunks[i], similarities[i]) for i in top_indices if similarities[i] > 0.1]
    
    except Exception as e:
        logging.error(f"Error in semantic search: {str(e)}")
        return []

def search_full_extracted_content(query):
    """
    Searches the full extracted content using semantic search.
    
    Args:
        query (str): User's query.

    Returns:
        str: Relevant sections from the full extracted content.
    """
    # Split text into paragraphs for searching
    paragraphs = [p for p in extracted_text.split("\n\n") if len(p.strip()) > 50]
    
    if not paragraphs:
        return ""
    
    # Perform semantic search
    relevant_results = semantic_search(query, paragraphs)
    
    # Combine results
    combined_results = "\n\n".join([f"Relevance {score:.2f}: {text}" for text, score in relevant_results])
    
    return combined_results

def search_sources(query):
    """
    Search both cleaned results and full extracted content.
    
    Args:
        query (str): The user's query
        
    Returns:
        dict: Relevant sections from both sources
    """
    relevant_sections = {}
    
    # Search in structured sections
    query_lower = query.lower()
    keywords = set(query_lower.split())
    
    # Check section relevance
    section_keywords = {
        "summary": {"summary", "overview", "abstract", "about"},
        "inputs": {"input", "inputs", "parameters", "variables", "data"},
        "outputs": {"output", "outputs", "results", "predictions"},
        "calculations": {"calculation", "calculations", "algorithm", "formula", "compute"},
        "model_performance": {"performance", "accuracy", "precision", "recall", "metrics"},
        "solution_specification": {"solution", "specification", "architecture", "design"},
        "testing_summary": {"testing", "test", "validation", "verify"},
        "reconciliation": {"reconciliation", "reconcile", "match", "compare"}
    }
    
    # Find relevant sections based on keyword overlap
    for section, section_keys in section_keywords.items():
        if keywords.intersection(section_keys) or any(key in query_lower for key in section_keys):
            if section in cleaned_results and cleaned_results[section]:
                # Map internal section names to display names
                display_name = section.replace("_", " ").title()
                relevant_sections[display_name] = cleaned_results[section]
    
    # If query mentions multiple topics or no specific sections matched, include summary
    if len(relevant_sections) == 0 or len(relevant_sections) > 3:
        if "summary" in cleaned_results and cleaned_results["summary"]:
            relevant_sections["Summary"] = cleaned_results["summary"]
    
    # Search in full extracted content
    additional_context = search_full_extracted_content(query)
    
    if additional_context:
        relevant_sections["Additional Context"] = additional_context
    
    return relevant_sections

def enhanced_semantic_search(query_dict):
    """
    Use transformed query for multi-strategy search approach.
    
    Args:
        query_dict (dict): The transformed query with various forms
        
    Returns:
        dict: Consolidated search results
    """
    # Search with primary query
    main_results = search_sources(query_dict["rephrased_query"])
    
    # If main results are insufficient, try related queries
    if not main_results or len(main_results) < 2:
        for related_query in query_dict["related_queries"][:2]:  # Try up to 2 related queries
            additional_results = search_sources(related_query)
            for section, content in additional_results.items():
                if section not in main_results:
                    main_results[section] = content
    
    return main_results

def format_content(content):
    """
    Format content for inclusion in prompts.
    
    Args:
        content: Content to format (could be string, list, or dict)
        
    Returns:
        str: Formatted content as string
    """
    if isinstance(content, list):
        if all(isinstance(item, dict) for item in content):
            # Handle structured data (like inputs/outputs)
            return "\n".join([f"- {item.get('name', 'Unknown')}: {item.get('description', 'No description')} ({item.get('format', 'Unknown format')})" for item in content])
        else:
            # Handle list of strings
            return "\n".join([f"- {item}" for item in content])
    else:
        # Handle string content
        return content

def refine_context(query, raw_context):
    """
    Use LLM to refine and optimize the context before generating the final answer.
    
    Args:
        query (str): The user's question
        raw_context (dict): The raw search results
        
    Returns:
        str: Refined and optimized context
    """
    # Convert raw context to text
    context_text = "\n\n".join([f"== {section} ==\n{format_content(content)}" 
                               for section, content in raw_context.items()])
    
    # Skip if context is small
    if len(context_text) < 2000:
        return context_text
        
    prompt = f"""
    You are an expert information analyst. I have a user question and some context that might help answer it.
    Please analyze this context and optimize it for answering the question.
    
    USER QUESTION: {query}
    
    CONTEXT:
    {context_text}
    
    Please:
    1. Remove any irrelevant portions that don't help answer the question
    2. Reorganize the remaining information in order of relevance to the question
    3. Preserve all section headings (== Section Name ==) and their structure
    4. Preserve all relevant data points, numbers, and specific details
    
    Return only the refined context, maintaining the section markers and formatting.
    """
    
    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000,
                "temperature": 0.2
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        refined_context = response_body["content"].strip()
        
        # If refinement somehow removed all section markers, revert to original
        if "==" not in refined_context:
            logging.warning("Context refinement removed section markers. Using original context.")
            return context_text
            
        return refined_context
    except Exception as e:
        logging.error(f"Error refining context: {str(e)}")
        return context_text

def generate_chain_of_thought_answer(query, context):
    """
    Generate answer using multi-step chain-of-thought reasoning.
    
    Args:
        query (str): The user's question
        context (str): The refined context
        
    Returns:
        str: Detailed answer with reasoning steps
    """
    prompt = f"""
    You are an expert answering questions about a financial model whitepaper.
    
    CONTEXT:
    {context}
    
    QUESTION: {query}
    
    Please answer this question following these steps:
    1. STEP 1 - ANALYSIS: Analyze what the question is asking and identify key information needed
    2. STEP 2 - EVIDENCE: Extract relevant evidence from the context
    3. STEP 3 - REASONING: Reason through the evidence step by step
    4. STEP 4 - CONCLUSION: Provide a final answer based on your reasoning
    
    Your response should show this structured thinking process, starting with "ANALYSIS:" and ending with "CONCLUSION:".
    If the context doesn't contain relevant information, state so in your analysis.
    """
    
    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 3000,
                "temperature": 0.5
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        detailed_answer = response_body["content"].strip()
        
        return detailed_answer
    except Exception as e:
        logging.error(f"Error generating detailed answer: {str(e)}")
        return f"Failed to generate answer: {str(e)}"

def evaluate_answer_quality(query, answer, context):
    """
    Use LLM to evaluate the quality of the generated answer.
    
    Args:
        query (str): The user's question
        answer (str): The generated answer
        context (str): The context used to generate the answer
        
    Returns:
        dict: Evaluation results with metrics and improvement suggestions
    """
    prompt = f"""
    You are an expert evaluator of question-answering systems.
    
    ORIGINAL QUESTION: {query}
    
    ANSWER PROVIDED:
    {answer}
    
    CONTEXT USED:
    {context}
    
    Please evaluate this answer on these criteria:
    1. Accuracy: Does the answer correctly reflect information in the context?
    2. Completeness: Does the answer address all aspects of the question?
    3. Conciseness: Is the answer appropriately detailed without unnecessary information?
    4. Citation: Does the answer properly cite sources from the context?
    
    For each criterion, rate from 1-5 where 5 is best.
    Also suggest how the answer could be improved.
    
    Format your response as JSON with these keys:
    "accuracy_score", "completeness_score", "conciseness_score", "citation_score", 
    "overall_score", "improvement_suggestions"
    """
    
    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        evaluation = json.loads(response_body["content"])
        
        # If quality is poor, log for review
        overall_score = evaluation.get("overall_score", 0)
        if overall_score < 3.5:
            logging.warning(f"Low quality answer detected (score: {overall_score})")
        
        return evaluation
    except Exception as e:
        logging.error(f"Error evaluating answer: {str(e)}")
        return {"overall_score": 3, "improvement_suggestions": "Could not evaluate answer."}

def generate_improved_answer(query, original_answer, evaluation):
    """
    If answer quality is low, use LLM to generate an improved version.
    
    Args:
        query (str): The user's question
        original_answer (str): The original answer
        evaluation (dict): Quality evaluation results
        
    Returns:
        str: Improved answer or original if improvement failed
    """
    # Only attempt improvement if score is below threshold and has suggestions
    if evaluation.get("overall_score", 5) >= 4 or not evaluation.get("improvement_suggestions"):
        return original_answer
        
    prompt = f"""
    You are an expert at improving answers to questions.
    
    ORIGINAL QUESTION: {query}
    
    ORIGINAL ANSWER:
    {original_answer}
    
    EVALUATION:
    The answer received these scores:
    - Accuracy: {evaluation.get("accuracy_score", "N/A")}/5
    - Completeness: {evaluation.get("completeness_score", "N/A")}/5
    - Conciseness: {evaluation.get("conciseness_score", "N/A")}/5
    - Citation: {evaluation.get("citation_score", "N/A")}/5
    
    IMPROVEMENT SUGGESTIONS:
    {evaluation.get("improvement_suggestions", "No specific suggestions.")}
    
    Please rewrite the answer addressing these improvement suggestions.
    Ensure the improved answer maintains any correct information from the original while fixing the issues identified.
    """
    
    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 3000,
                "temperature": 0.3
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        improved_answer = response_body["content"].strip()
        
        logging.info("Generated improved answer based on evaluation")
        return improved_answer
    except Exception as e:
        logging.error(f"Error improving answer: {str(e)}")
        return original_answer  # Fallback to original answer

def generate_followup_questions(query, answer):
    """
    Generate relevant follow-up questions based on the question and answer.
    
    Args:
        query (str): The original user question
        answer (str): The answer provided
        
    Returns:
        list: List of suggested follow-up questions
    """
    prompt = f"""
    You are an expert at identifying insightful follow-up questions.
    
    ORIGINAL QUESTION: {query}
    
    ANSWER PROVIDED:
    {answer}
    
    Please suggest 3 natural follow-up questions that:
    1. Explore related aspects not covered in the original answer
    2. Dig deeper into specifics mentioned in the answer
    3. Clarify potential ambiguities or explore edge cases
    
    Format your response as a JSON array of strings, each containing one follow-up question.
    """
    
    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.7,
                "response_format": {"type": "json_object"}
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        followups = json.loads(response_body["content"])
        
        return followups if isinstance(followups, list) else []
    except Exception as e:
        logging.error(f"Error generating follow-up questions: {str(e)}")
        return []

def answer_question(user_query):
    """
    Full pipeline for answering questions with enhanced features.
    
    Args:
        user_query (str): The user's question
        
    Returns:
        dict: Complete response package with answer, evaluation, and follow-ups
    """
    # Step 1: Query analysis and transformation
    query_dict = analyze_and_transform_query(user_query)
    
    # Step 2: Enhanced semantic search
    search_results = enhanced_semantic_search(query_dict)
    
    if not search_results:
        return {
            "answer": "I couldn't find relevant information to answer this question.",
            "confidence": 0,
            "followup_questions": [],
            "sources": []
        }
    
    # Step 3: Context refinement
    refined_context = refine_context(user_query, search_results)
    
    # Step 4: Chain-of-thought answer generation
    detailed_answer = generate_chain_of_thought_answer(user_query, refined_context)
    
    # Step 5: Answer quality evaluation
    evaluation = evaluate_answer_quality(user_query, detailed_answer, refined_context)
    
    # Step 6: Answer improvement (if needed)
    final_answer = generate_improved_answer(user_query, detailed_answer, evaluation) \
                  if evaluation.get("overall_score", 5) < 4 else detailed_answer
    
    # Step 7: Generate follow-up questions
    followup_questions = generate_followup_questions(user_query, final_answer)
    
    # Cache the result
    result = {
        "answer": final_answer,
        "confidence": evaluation.get("overall_score", 3),
        "followup_questions": followup_questions,
        "sources": list(search_results.keys())
    }
    
    answer_cache[user_query] = result
    return result

def run_qa_cli():
    """Run a simple command-line interface for the Q&A system."""
    global cleaned_results, extracted_text, bedrock_client
    
    # File paths - adjust these to match your actual file locations
    cleaned_results_path = "./cleaned_whitepaper_analysis.json"
    extracted_text_path = "./extracted_content/extracted_text.txt"
    
    # Load data and set up bedrock client
    cleaned_results, extracted_text = load_sources(cleaned_results_path, extracted_text_path)
    bedrock_client = setup_bedrock_client()
    
    print("\n=== Whitepaper Q&A System ===")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        query = input("\nEnter your question: ").strip()
        
        if query.lower() in ['exit', 'quit']:
            print("Exiting Q&A system. Goodbye!")
            break
        
        if not query:
            print("Please enter a valid question.")
            continue
        
        print("\nProcessing your question...\n")
        
        try:
            # Process the question
            result = answer_question(query)
            
            # Display answer with confidence
            print("\n=== ANSWER ===")
            print(f"Confidence: {result.get('confidence', 0)}/5")
            print(result['answer'])
            
            # Display sources
            if result.get("sources"):
                print("\n=== SOURCES ===")
                for source in result["sources"]:
                    print(f"- {source}")
            
            # Display follow-up questions
            if result.get("followup_questions"):
                print("\n=== FOLLOW-UP QUESTIONS ===")
                for i, q in enumerate(result["followup_questions"], 1):
                    print(f"{i}. {q}")
            
        except Exception as e:
            print(f"Error processing question: {str(e)}")
        
        print("\n" + "-"*60)

# Run the system
if __name__ == "__main__":
    run_qa_cli()






#How to Run This Simplified Code

cleaned_results = None  # Will store your cleaned whitepaper analysis
extracted_text = None   # Will store your full extracted text
bedrock_client = None   # Will connect to AWS Bedrock
answer_cache = {}       # Will store previous answers



Step-by-Step Explanation
1. Setting Up
The first part of the code sets up logging and global variables:

python
cleaned_results = None  # Will store your cleaned whitepaper analysis
extracted_text = None   # Will store your full extracted text
bedrock_client = None   # Will connect to AWS Bedrock
answer_cache = {}       # Will store previous answers
2. Loading Data
The load_sources function loads your data files:

python
# Load the cleaned summary and full text
cleaned_results, extracted_text = load_sources(cleaned_results_path, extracted_text_path)
3. Setting Up AWS
The setup_bedrock_client function connects to AWS:

python
# Connect to AWS Bedrock for Claude access
bedrock_client = setup_bedrock_client()
4. Question Flow
When you ask a question, it follows these steps:

python
# Step 1: Analyze the question to understand it better
query_dict = analyze_and_transform_query(user_query)

# Step 2: Search for relevant information
search_results = enhanced_semantic_search(query_dict)

# Step 3: Clean up and focus the information
refined_context = refine_context(user_query, search_results)

# Step 4: Generate a detailed answer
detailed_answer = generate_chain_of_thought_answer(user_query, refined_context)

# Step 5: Check if the answer is good
evaluation = evaluate_answer_quality(user_query, detailed_answer, refined_context)

# Step 6: Improve the answer if needed
if evaluation score < 4:
    final_answer = generate_improved_answer(user_query, detailed_answer, evaluation)
else:
    final_answer = detailed_answer

# Step 7: Suggest follow-up questions
followup_questions = generate_followup_questions(user_query, final_answer)

# Return the results
return {answer, confidence, followups, sources}

