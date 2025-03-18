# preprocessing_whitepaper.py
import json
import os
import re
from PyPDF2 import PdfReader

def preprocess_pdf(file_path, chunk_size=10000):
    """Extracts text from a PDF, divides it into chunks."""
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            all_chunks = []
            current_chunk = ""

            for page in reader.pages:
                text = page.extract_text()
                text = re.sub(r'\s+', ' ', text).strip()  # Clean whitespace
                current_chunk += text + " "

                while len(current_chunk) >= chunk_size:
                    split_index = current_chunk.rfind('. ', 0, chunk_size)
                    if split_index == -1:
                        split_index = chunk_size
                    chunk_text = current_chunk[:split_index].strip()
                    all_chunks.append({"text": chunk_text})
                    current_chunk = current_chunk[split_index:].strip()

            if current_chunk:
                all_chunks.append({"text": current_chunk.strip()})
            return all_chunks

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_whitepaper(input_dir, output_dir):
    """Processes only the whitepaper.pdf."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(input_dir, 'whitepaper.pdf') # Only whitepaper.pdf
    if not os.path.exists(file_path):
        print(f"Error: whitepaper.pdf not found in {input_dir}")
        return

    chunks = preprocess_pdf(file_path)

    if chunks:
        for i, chunk in enumerate(chunks):
            output_file_path = os.path.join(output_dir, f"whitepaper_chunk_{i}.json")
            with open(output_file_path, 'w') as outfile:
                json.dump(chunk, outfile, indent=4)
            print(f"Processed chunk {i}, saved to {output_file_path}")

# --- Example Usage ---
if __name__ == '__main__':
    input_directory = 'input_pdfs'
    output_directory = 'preprocessed_pdfs'
    if not os.path.exists(input_directory):
      os.makedirs(input_directory)

    # --- (Optional: Create a Dummy whitepaper.pdf for testing) ---
    from reportlab.pdfgen import canvas
    def create_dummy_pdf(filepath, content):
        c = canvas.Canvas(filepath)
        c.drawString(100, 750, content)
        c.save()

    if not os.path.exists(os.path.join(input_directory, "whitepaper.pdf")):
      create_dummy_pdf(os.path.join(input_directory, "whitepaper.pdf"), "This is a dummy whitepaper for the Loan Risk Model (LRM) version 1.0. Inputs are Borrower Annual Income (float), Loan-to-Value Ratio (float), and Credit History Length (integer). The output is a Predicted Default Probability (float). The model performance is measured by AUC (0.85).")

    process_whitepaper(input_directory, output_directory)
  
