!pip install boto3 scikit-learn ipywidgets


import os
import json
import logging
import boto3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedLLMQASystem:
    def __init__(self, cleaned_results_path, extracted_text_path):
        """Initialize the advanced LLM-powered Q&A system."""
        self.cleaned_results, self.extracted_text = self._load_sources(
            cleaned_results_path, extracted_text_path)
        self.bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.query_history = []
        self.answer_cache = {}
        
    def _load_sources(self, cleaned_file_path, extracted_file_path):
        """Load source data."""
        cleaned_results, extracted_text = None, None
        
        try:
            if os.path.exists(cleaned_file_path):
                with open(cleaned_file_path, "r", encoding="utf-8") as f:
                    cleaned_results = json.load(f)
                logging.info(f"Loaded cleaned results from {cleaned_file_path}")
            else:
                logging.warning(f"Cleaned results file not found: {cleaned_file_path}")
            
            if os.path.exists(extracted_file_path):
                with open(extracted_file_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
                logging.info(f"Loaded extracted text from {extracted_file_path}")
            else:
                logging.warning(f"Extracted text file not found: {extracted_file_path}")
            
            return cleaned_results, extracted_text
        except Exception as e:
            logging.error(f"Error loading sources: {str(e)}")
            return None, None

    def analyze_and_transform_query(self, user_query):
        """Use LLM to analyze the query and transform it into optimal search queries."""
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
            response = self.bedrock_client.invoke_model(
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

    def _search_sources(self, query):
        """
        Search both cleaned results and full extracted content.
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
                if section in self.cleaned_results and self.cleaned_results[section]:
                    # Map internal section names to display names
                    display_name = section.replace("_", " ").title()
                    relevant_sections[display_name] = self.cleaned_results[section]
        
        # If query mentions multiple topics or no specific sections matched, include summary
        if len(relevant_sections) == 0 or len(relevant_sections) > 3:
            if "summary" in self.cleaned_results and self.cleaned_results["summary"]:
                relevant_sections["Summary"] = self.cleaned_results["summary"]
        
        # Search in full extracted content
        additional_context = self._search_full_extracted_content(query)
        
        if additional_context:
            relevant_sections["Additional Context"] = additional_context
        
        return relevant_sections

    def _search_full_extracted_content(self, query):
        """
        Searches the full extracted content using semantic search.
        """
        # Split text into paragraphs for searching
        paragraphs = [p for p in self.extracted_text.split("\n\n") if len(p.strip()) > 50]
        
        if not paragraphs:
            return ""
        
        # Perform semantic search
        relevant_results = self._semantic_search(query, paragraphs)
        
        # Combine results
        combined_results = "\n\n".join([f"Relevance {score:.2f}: {text}" for text, score in relevant_results])
        
        return combined_results

    def _semantic_search(self, query, text_chunks, top_k=5):
        """
        Performs semantic search using TF-IDF and cosine similarity.
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

    def enhanced_semantic_search(self, query_dict):
        """
        Use transformed query for multi-strategy search approach.
        """
        # Search with primary query
        main_results = self._search_sources(query_dict["rephrased_query"])
        
        # If main results are insufficient, try related queries
        if not main_results or len(main_results) < 2:
            for related_query in query_dict["related_queries"][:2]:  # Try up to 2 related queries
                additional_results = self._search_sources(related_query)
                for section, content in additional_results.items():
                    if section not in main_results:
                        main_results[section] = content
                        
        return main_results

    def _format_content(self, content):
        """Format content for inclusion in prompts."""
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

    def refine_context(self, query, raw_context):
        """
        Use LLM to refine and optimize the context before generating the final answer.
        """
        # Convert raw context to text
        context_text = "\n\n".join([f"== {section} ==\n{self._format_content(content)}" 
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
            response = self.bedrock_client.invoke_model(
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

    def generate_chain_of_thought_answer(self, query, context):
        """
        Generate answer using multi-step chain-of-thought reasoning.
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
            response = self.bedrock_client.invoke_model(
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

    def evaluate_answer_quality(self, query, answer, context):
        """
        Use LLM to evaluate the quality of the generated answer.
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
            response = self.bedrock_client.invoke_model(
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

    def generate_improved_answer(self, query, original_answer, evaluation):
        """
        If answer quality is low, use LLM to generate an improved version.
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
            response = self.bedrock_client.invoke_model(
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

    def generate_followup_questions(self, query, answer):
        """
        Generate relevant follow-up questions based on the question and answer.
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
            response = self.bedrock_client.invoke_model(
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

    def answer_question(self, user_query):
        """
        Full LLM-powered pipeline for answering questions with enhanced features.
        """
        # Step 1: Query analysis and transformation
        query_dict = self.analyze_and_transform_query(user_query)
        
        # Step 2: Enhanced semantic search
        search_results = self.enhanced_semantic_search(query_dict)
        
        if not search_results:
            return {
                "answer": "I couldn't find relevant information to answer this question.",
                "confidence": 0,
                "followup_questions": []
            }
        
        # Step 3: Context refinement
        refined_context = self.refine_context(user_query, search_results)
        
        # Step 4: Chain-of-thought answer generation
        detailed_answer = self.generate_chain_of_thought_answer(user_query, refined_context)
        
        # Step 5: Answer quality evaluation
        evaluation = self.evaluate_answer_quality(user_query, detailed_answer, refined_context)
        
        # Step 6: Answer improvement (if needed)
        final_answer = self.generate_improved_answer(user_query, detailed_answer, evaluation) \
                      if evaluation.get("overall_score", 5) < 4 else detailed_answer
        
        # Step 7: Generate follow-up questions
        followup_questions = self.generate_followup_questions(user_query, final_answer)
        
        # Cache the result
        self.answer_cache[user_query] = {
            "answer": final_answer,
            "confidence": evaluation.get("overall_score", 3),
            "followup_questions": followup_questions,
            "sources": list(search_results.keys())
        }
        
        return self.answer_cache[user_query]

# Command-line interface
def cli_qa(qa_system):
    """Simple command-line interface for the Q&A system."""
    print("\n=== Advanced LLM-Powered Whitepaper Q&A System ===")
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
            # Process with full LLM pipeline
            result = qa_system.answer_question(query)
            
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

def main():
    """Main function to run the Q&A system."""
    # File paths
    cleaned_results_path = "./cleaned_whitepaper_analysis.json"
    extracted_text_path = "./extracted_content/extracted_text.txt"
    
    # Initialize the system
    qa_system = AdvancedLLMQASystem(cleaned_results_path, extracted_text_path)
    
    # Run the CLI interface
    cli_qa(qa_system)

if __name__ == "__main__":
    main()









from advanced_qa_system import AdvancedLLMQASystem

# Initialize the system
qa_system = AdvancedLLMQASystem(
    "./cleaned_whitepaper_analysis.json",
    "./extracted_content/extracted_text.txt"
)

# Test query analysis
query = "What are the key inputs to the model?"
query_dict = qa_system.analyze_and_transform_query(query)
print("Transformed Query:", query_dict)

# Test search
search_results = qa_system.enhanced_semantic_search(query_dict)
print("Search Results:", search_results.keys())

# Test context refinement
refined_context = qa_system.refine_context(query, search_results)
print("Refined Context Length:", len(refined_context))

# Test answer generation
answer = qa_system.generate_chain_of_thought_answer(query, refined_context)
print("Generated Answer:", answer[:100] + "...")

# Test full pipeline
result = qa_system.answer_question(query)
print("Final Answer:", result["answer"])
print("Confidence:", result["confidence"])
print("Follow-up Questions:", result["followup_questions"])





#For an interactive interface in Jupyter, create a file named qa_notebook.ipynb:



import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from advanced_qa_system import AdvancedLLMQASystem

# Initialize the system
qa_system = AdvancedLLMQASystem(
    "./cleaned_whitepaper_analysis.json",
    "./extracted_content/extracted_text.txt"
)

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
        
        print(f"Question: {query}\n")
        print("Processing...")
        
        # Process with full LLM pipeline
        result = qa_system.answer_question(query)
        
        # Display answer with confidence
        print("\nAnswer:")
        print(f"[Confidence: {result.get('confidence', 0)}/5]")
        print(result['answer'])
        
        # Display sources
        if result.get("sources"):
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source}")
        
        # Display follow-up questions
        if result.get("followup_questions"):
            print("\nFollow-up questions you might ask:")
            for i, q in enumerate(result["followup_questions"], 1):
                print(f"{i}. {q}")

submit_button.on_click(on_submit_button_clicked)

# Display interface
display(widgets.HBox([question_input, submit_button]))
display(output_area)
