import zipfile
import os

zip_file_path = '/content/release-raw.zip'

extract_to = '/content/release-raw/'

os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"Files extracted to {extract_to}")
