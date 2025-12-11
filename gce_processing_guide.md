# Guide: Automating Financial Signal Processing with Google Compute Engine

This guide provides a step-by-step walkthrough for setting up a Google Compute Engine (GCE) virtual machine to automatically run the financial analysis script (`boll4-nov-g.py`), generate JSON output, and make it available for ingestion into a data pipeline.

## Introduction

The goal is to create an automated, server-based environment for processing the 200+ trading signals. By running the script on a GCE VM, we can schedule its execution, manage its environment securely, and use its output to trigger downstream processes, forming a complete data pipeline. The script will generate a detailed JSON file and upload it to a Google Cloud Storage (GCS) bucket, from where it can be consumed by other services.

## Prerequisites

1.  **Google Cloud Account:** You need an active GCP account with billing enabled.
2.  **gcloud CLI:** The [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) must be installed and authenticated (`gcloud auth login`).
3.  **Project Files:** You should have the `boll4-nov-g.py` script and a `requirements.txt` file ready.

---

## Step 1: Prepare the Python Environment

For the script to run on the VM, its dependencies must be defined. Create a file named `requirements.txt` in the same directory as your Python script with the following content:

**`requirements.txt`**
```
yfinance
pandas
numpy
google-cloud-storage
google-genai
python-dotenv
```

---

## Step 2: Create a Google Compute Engine (GCE) VM

We will create a small, cost-effective VM. The `e2-micro` is a good choice for this task as it's part of the free tier (subject to GCP's terms). We'll grant it the necessary permissions to access other Google Cloud services.

Open your terminal and run the following `gcloud` command:

```bash
gcloud compute instances create signal-processor-vm \
    --project=your-gcp-project-id \
    --zone=us-central1-a \
    --machine-type=e2-micro \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --scopes=https://www.googleapis.com/auth/cloud-platform
```

**Command Breakdown:**
*   `signal-processor-vm`: The name of our new VM.
*   `--project`: Your GCP project ID.
*   `--zone`: The GCP zone to create the VM in. You can choose one closer to you.
*   `--machine-type`: `e2-micro` is a small, shared-core machine suitable for this task.
*   `--image-family` & `--image-project`: Specifies the operating system. We're using Debian 11.
*   `--scopes`: **This is critical.** It gives the VM the necessary IAM permissions to access GCP services like Google Cloud Storage.

---

## Step 3: Deploy the Script to the VM

Next, we need to copy our project files to the VM.

1.  **Use `gcloud compute scp` to copy your files:**
    This command securely copies files to the VM's home directory.

    ```bash
    # Make sure you are in the directory containing the files
    gcloud compute scp boll4-nov-g.py requirements.txt signal-processor-vm:~/ --zone=us-central1-a
    ```

2.  **SSH into the VM to configure it:**

    ```bash
    gcloud compute ssh signal-processor-vm --zone=us-central1-a
    ```

3.  **Inside the VM**, install the required software and Python packages:

    ```bash
    # Update package lists
    sudo apt-get update

    # Install Python and pip
    sudo apt-get install python3-pip -y

    # Install the script's dependencies
    pip3 install -r requirements.txt
    ```

---

## Step 4: Run the Script Manually

Before automating, it's good practice to run the script once manually to ensure everything works.

1.  **Set the API Key Securely:**
    Inside the VM's SSH session, export the `GEMINI_API_KEY` as an environment variable. This is more secure than placing it in a file on the server.

    ```bash
    export GEMINI_API_KEY="your-api-key-here"
    ```

2.  **Run the script:**

    ```bash
    python3 boll4-nov-g.py
    ```

The script should now execute, process the signals, and upload the resulting JSON file to your Google Cloud Storage bucket as configured within the script.

---

## Step 5: Automate with a Startup Script

To run the analysis automatically (e.g., daily), you can use a startup script. This script runs every time the VM boots up.

1.  **Create a `startup-script.sh` file on your local machine.**
    This script will pull the latest code from a Git repository, install dependencies, retrieve the API key from **Secret Manager** (the recommended best practice), and run the analysis.

    **`startup-script.sh`**
    ```bash
    #!/bin/bash

    # Wait for network to be ready
    sleep 10

    # Install dependencies
    sudo apt-get update
    sudo apt-get install -y python3-pip git

    # Clone or pull the repository
    cd /home/your_user
    if [ -d "alpha-fullstack" ] ; then
      cd alpha-fullstack
      git pull
    else
      git clone https://github.com/your-repo/alpha-fullstack.git
      cd alpha-fullstack
    fi

    # Install Python packages
    pip3 install -r ai-fin3/requirements.txt

    # --- Best Practice: Get API Key from Secret Manager ---
    # 1. Create a secret in GCP Secret Manager named "gemini-api-key"
    # 2. Ensure your VM's service account has the "Secret Manager Secret Accessor" role.
    export GEMINI_API_KEY=$(gcloud secrets versions access latest --secret="gemini-api-key")

    # Run the Python script
    cd ai-fin3
    python3 boll4-nov-g.py

    # Optional: Shut down the VM to save costs after the script runs
    # The VM will run the script again the next time it's started.
    sudo poweroff
    ```

2.  **Add the startup script to your VM.**
    You can do this when creating the VM or add it to an existing one.

    ```bash
    # Add to an existing VM
    gcloud compute instances add-metadata signal-processor-vm \
        --metadata-from-file startup-script=startup-script.sh \
        --zone=us-central1-a
    ```

Now, every time you start the VM, it will automatically run the latest version of your analysis script. You can use Cloud Scheduler to start and stop the VM on a daily schedule.

---

## Step 6: Ingesting the JSON to Complete the Pipeline

The script is designed to save its JSON output to a Google Cloud Storage (GCS) bucket. This is the handoff point for the rest of your pipeline.

**Pipeline Trigger:**
The most effective way to "ingest" the JSON is to use a **GCS trigger** for a **Cloud Function**.

1.  **Create a Cloud Function:** In the GCP Console, create a new Cloud Function.
2.  **Set the Trigger:** Configure the trigger to be `Google Cloud Storage` and the event type to be `google.storage.object.finalize`.
3.  **Select the Bucket:** Choose the GCS bucket where your script saves the JSON files.
4.  **Write the Function Code:** The Cloud Function will receive information about the newly created JSON file. You can write code (e.g., in Python or Node.js) to:
    *   Read the JSON file from GCS.
    *   Parse its content.
    *   Insert the data into a database (like BigQuery or Firestore).
    *   Send a notification.
    *   Trigger another data processing job.

This event-driven architecture creates a seamless, serverless pipeline that automatically processes the data as soon as it's generated by your GCE script.

```