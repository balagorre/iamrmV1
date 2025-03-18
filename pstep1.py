# Upload PDF to S3
s3.upload_file(pdf_path, bucket_name, s3_key)

# Start Textract job
response = textract.start_document_analysis(
    DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': s3_key}},
    FeatureTypes=['TABLES', 'FORMS']
)

# Retrieve results
results = []
while True:
    response = textract.get_document_analysis(JobId=job_id)
    results.append(response)
    next_token = response.get('NextToken')
    if not next_token:
        break

# Save results
with open("textract_results.json", "w") as f:
    json.dump(results, f)
