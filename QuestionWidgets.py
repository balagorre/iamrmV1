










def analyze_whitepaper(embeddings_db):
    """Generate a structured analysis of the whitepaper"""
    key_sections = [
        "What are the key inputs required by this model?",
        "What are the main outputs produced by this model?",
        "What are the key assumptions made in this model?",
        "What are the limitations or constraints of this model?",
        "How is model performance evaluated or validated?",
        "What calculations or algorithms are central to this model?"
    ]
    
    analysis = {}
    
    print("Analyzing whitepaper...")
    for question in key_sections:
        print(f"\nAnalyzing: {question}")
        analysis[question] = ask_question(question, embeddings_db)
        time.sleep(2)  # Prevent rate limiting
    
    # Save analysis
    with open(f"{output_dir}/whitepaper_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    
    print("\nWhitepaper analysis complete and saved.")
    return analysis

# Generate comprehensive analysis
whitepaper_analysis = analyze_whitepaper(embeddings_db)



import ipywidgets as widgets
from IPython.display import display, clear_output

def interactive_qa():
    """Create an interactive Q&A interface"""
    # Load embeddings if not in memory
    try:
        with open(f"{output_dir}/embeddings_db.json", "r") as f:
            embeddings_db = json.load(f)
    except:
        print("Error: Could not load embeddings database")
        return
    
    # Create widgets
    question_input = widgets.Text(
        value='',
        placeholder='Ask a question about the whitepaper',
        description='Question:',
        disabled=False,
        layout=widgets.Layout(width='80%')
    )
    
    submit_button = widgets.Button(
        description='Ask',
        disabled=False,
        button_style='primary',
        tooltip='Submit question',
        icon='question'
    )
    
    output_area = widgets.Output()
    
    # Define button click event
    def on_submit_button_clicked(b):
        with output_area:
            clear_output()
            if question_input.value:
                ask_question(question_input.value, embeddings_db)
            else:
                print("Please enter a question.")
    
    submit_button.on_click(on_submit_button_clicked)
    
    # Display interface
    display(widgets.HBox([question_input, submit_button]))
    display(output_area)

# Create interactive interface
interactive_qa()









import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown
import json

output_dir = "./data"

def interactive_qa():
    # Load embeddings database
    try:
        with open(f"{output_dir}/embeddings_db.json", "r") as f:
            embeddings_db = json.load(f)
    except Exception as e:
        display(Markdown(f"**Error:** `{e}`"))
        return

    # Question input widget
    question_input = widgets.Text(
        placeholder='Type your question here...',
        description='Question:',
        layout=widgets.Layout(width='80%')
    )

    # Submit button
    submit_button = widgets.Button(
        description='Ask',
        button_style='success',
        icon='question-circle'
    )

    # Function to format and display answer
    def ask_question(question, embeddings_db):
        # Replace with actual query logic
        response = f"Here is a detailed answer related to your question: '{question}'."

        markdown_response = (
            f"---\n"
            f"### **Your Question:**\n"
            f"> {question}\n\n"
            f"### **Detailed Answer:**\n"
            f"{response}\n\n"
            f"---\n"
        )

        display(Markdown(markdown_response))

    # Handle button click
    def on_submit_button_clicked(b):
        with clear_output(wait=True):
            if question_input.value.strip():
                ask_question(question_input.value, embeddings_db)
            else:
                display(Markdown("*Please enter a valid question before submitting.*"))

    submit_button.on_click(on_submit_button_clicked)

    # Display widgets
    display(widgets.VBox([
        widgets.HBox([question_input, submit_button]),
    ]))

# Run the interactive widget
interactive_qa()









def on_submit_button_clicked(b):
    with output_area:
        clear_output(wait=True)
        query = question_input.value.strip()
        if not query:
            display(Markdown("<span style='color:#C0392B'>*Please enter a valid question.*</span>"))
            return

        display(Markdown(f"### <span style='color:#003366'>**Question:**</span>\n> {query}\n\n<span style='color:#17A589'>*Processing...*</span>"))

        # Process with full LLM pipeline
        result = qa_system.answer_question(query)

        markdown_response = (
            f"### <span style='color:#154360'>**Answer:**</span>\n"
            f"<span style='color:#2874A6'>*[Confidence: {result.get('confidence', 0)}/5]*</span>\n\n"
            f"{result['answer']}\n"
        )

        # Display sources
        if result.get("sources"):
            markdown_response += "\n### <span style='color:#117864'>**Sources:**</span>\n"
            for source in result["sources"]:
                markdown_response += f"- {source}\n"

        # Display follow-up questions
        if result.get("followup_questions"):
            markdown_response += "\n### <span style='color:#AF7AC5'>**Follow-up Questions:**</span>\n"
            for i, q in enumerate(result["followup_questions"], 1):
                markdown_response += f"{i}. {q}\n"

        display(Markdown(markdown_response))



