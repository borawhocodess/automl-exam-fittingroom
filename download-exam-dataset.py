import os
import requests
import zipfile
import io

def download_and_unzip(url, extract_to):
    """
    Download a zip file from the specified URL and extract its contents directly
    to a folder, without saving the zip file to disk.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    zip_bytes = io.BytesIO()
    for chunk in response.iter_content(chunk_size=8192):
        zip_bytes.write(chunk)

    zip_bytes.seek(0)

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"Downloaded and extracted zip file from {url} to: {extract_to}")

def main():
    # URL for the exam dataset zip
    url_exam = "https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-tabular/tabular-phase2.zip"

    extract_folder = "data"

    download_and_unzip(url_exam, extract_folder)

if __name__ == "__main__":
    main()