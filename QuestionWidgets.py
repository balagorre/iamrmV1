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
