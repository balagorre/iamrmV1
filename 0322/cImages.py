import fitz  # PyMuPDF

def extract_images_from_pdf(pdf_path: str, output_dir: str):
    doc = fitz.open(pdf_path)
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            with open(f"{output_dir}/page{page_number+1}_img{img_index+1}.{image_ext}", "wb") as f:
                f.write(image_bytes)
    print("✅ Image extraction complete.")
