"""
GCP-ONLY SCRIPT
All non‑GCP code has been removed.
This file now contains ONLY logic related to Google Cloud Storage
and local → GCS upload flows using GCP CLI authentication (ADC).
"""

import os
from pathlib import Path
from google.cloud import storage
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
LOCAL_SAVE_DIR = Path("./spreadsyo")
LOCAL_SAVE_DIR.mkdir(exist_ok=True)

BUCKET_NAME = os.environ.get("GCP_BUCKET", "ttb-1")

# -----------------------------
# INIT (GCP)
# -----------------------------
# Uses Application Default Credentials from:
#   gcloud auth application-default login
storage_client = storage.Client()

# -----------------------------
# LOCAL FILE HELPERS
# -----------------------------

def save_locally(filename: str, content: str) -> Path:
    """Save a file locally before uploading to GCS. This is strictly part of the GCP pipeline."""
    output_path = LOCAL_SAVE_DIR / filename
    output_path.write_text(content)
    return output_path

# -----------------------------
# GCS UPLOAD HELPERS
# -----------------------------

def upload_to_gcs(local_path: Path, remote_path: str):
    """Uploads a local file to GCS using ADC credentials."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(remote_path)
    blob.upload_from_filename(str(local_path))
    return f"gs://{BUCKET_NAME}/{remote_path}"


def upload_missing_files(local_folder: str, bucket_name: str):
    """
    Uploads all files from a folder to GCS ONLY if they don't already exist.
    """
    bucket = storage_client.bucket(bucket_name)
    existing_blobs = {blob.name for blob in bucket.list_blobs()}

    local_folder_path = Path(local_folder)
    if not local_folder_path.exists():
        print(f"Folder '{local_folder}' does not exist.")
        return

    for file_path in local_folder_path.glob("*"):
        if file_path.is_file():
            remote_name = file_path.name

            if remote_name in existing_blobs:
                print(f"SKIP: {remote_name} already exists in bucket.")
                continue

            blob = bucket.blob(remote_name)
            blob.upload_from_filename(str(file_path))
            print(f"UPLOADED: {remote_name}")

    print("Done: Missing files uploaded.")

# -----------------------------
# EXAMPLE EXECUTION
# -----------------------------

if __name__ == "__main__":
    # Example of saving a test file locally
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"example_{timestamp}.txt"
    local_file = save_locally(filename, "Sample GCP-only file content.")

    # Upload to GCS
    gcs_uri = upload_to_gcs(local_file, f"examples/{filename}")
    print("Uploaded to:", gcs_uri)

    # Sync any missing files
    upload_missing_files("./local_outputs", BUCKET_NAME)
