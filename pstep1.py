import fitz  # PyMuPDF
import tabula
import pytesseract
from PIL import Image
import os
import json

def extract_text_from_pdf(pdf_path, output_dir):
    """
    Extracts text from a PDF file using PyMuPDF (fitz).
    Saves extracted text to a file.
    """
    os.makedirs(output_dir, exist_ok=True)
    extracted_text = ""
    
    # Open the PDF with PyMuPDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Processing {total_pages} pages from {pdf_path}...")
    
    for page_num in range(total_pages):
        page = doc[page_num]
        page_text = page.get_text()
        extracted_text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
        
        # Save individual page text for debugging/inspection
        with open(f"{output_dir}/page_{page_num + 1:03d}.txt", "w", encoding="utf-8") as f:
            f.write(page_text)
    
    # Save full extracted text
    full_text_path = os.path.join(output_dir, "extracted_text.txt")
    with open(full_text_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    print(f"Extracted text saved to {full_text_path}")
    return extracted_text

def extract_tables_from_pdf(pdf_path, output_dir):
    """
    Extracts tables from a PDF file using Tabula.
    Saves each table as a CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Extract tables from all pages
        tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True, lattice=True)
        
        if not tables:
            print("No tables found in the document.")
            return []
        
        table_files = []
        for i, table in enumerate(tables):
            if not table.empty:
                table_file = os.path.join(output_dir, f"table_{i + 1}.csv")
                table.to_csv(table_file, index=False)
                table_files.append(table_file)
                print(f"Table {i + 1} saved to {table_file}")
        
        return table_files
    except Exception as e:
        print(f"Error extracting tables: {e}")
        return []

def extract_images_from_pdf(pdf_path, output_dir):
    """
    Extracts images from a PDF file using PyMuPDF (fitz).
    Applies OCR to images if needed.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    image_files = []
    
    for page_num in range(total_pages):
        page = doc[page_num]
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Save image file
            image_file = os.path.join(output_dir, f"page_{page_num + 1}_image_{img_index + 1}.{image_ext}")
            with open(image_file, "wb") as f:
                f.write(image_bytes)
            
            image_files.append(image_file)
            
            # Apply OCR to the image if needed
            try:
                img = Image.open(image_file)
                ocr_text = pytesseract.image_to_string(img)
                
                if ocr_text.strip():
                    ocr_file = image_file.replace(f".{image_ext}", "_ocr.txt")
                    with open(ocr_file, "w", encoding="utf-8") as f:
                        f.write(ocr_text)
                    print(f"OCR applied to {image_file}, text saved to {ocr_file}")
            except Exception as e:
                print(f"Error applying OCR to {image_file}: {e}")
    
    print(f"Extracted {len(image_files)} images.")
    return image_files

def process_local_pdf(pdf_path, output_dir):
    """
    Full workflow to process a local PDF file.
    Extracts text, tables, and images.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\nStep 1: Extracting text...")
    extracted_text = extract_text_from_pdf(pdf_path, os.path.join(output_dir, "text"))

    print("\nStep 2: Extracting tables...")
    table_files = extract_tables_from_pdf(pdf_path, os.path.join(output_dir, "tables"))

    print("\nStep 3: Extracting images...")
    image_files = extract_images_from_pdf(pdf_path, os.path.join(output_dir, "images"))

    # Save metadata about extracted content
    metadata = {
        "pdf_path": pdf_path,
        "total_pages": len(fitz.open(pdf_path)),
        "text_file": os.path.join(output_dir, "text/extracted_text.txt"),
        "table_files": table_files,
        "image_files": image_files,
    }

    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nMetadata saved to {metadata_file}")
    
# Example usage
pdf_path = "./data/model_whitepaper.pdf"
output_dir = "./processed_content"

process_local_pdf(pdf_path, output_dir)
