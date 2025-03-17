import PyPDF2
import pdfplumber
import csv

def extract_text_from_pdf(pdf_path, output_txt_path):
    """
    Extracts text from a PDF and saves it to a text file.
    Also extracts tabular data using pdfplumber and saves it as a CSV.
    
    :param pdf_path: Path to the PDF file.
    :param output_txt_path: Path to save extracted text.
    """
    try:
        # Extract text using PyPDF2
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text successfully extracted and saved to {output_txt_path}")
        
        # Extract tables using pdfplumber
        table_data = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_table()
                if tables:
                    table_data.extend(tables)
        
        # Save extracted tables to CSV
        if table_data:
            csv_output_path = output_txt_path.replace(".txt", ".csv")
            with open(csv_output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(table_data)
            print(f"Table data extracted and saved to {csv_output_path}")
        else:
            print("No tabular data found in the PDF.")
        
    except Exception as e:
        print(f"Error extracting data: {e}")

# Example usage
pdf_file = "your_model_whitepaper.pdf"  # Replace with your actual file
output_txt_file = "extracted_text.txt"
extract_text_from_pdf(pdf_file, output_txt_file)
